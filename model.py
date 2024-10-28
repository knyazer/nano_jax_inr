import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.spatial
from absl import logging
from jaxtyping import Array, Float, Int, PRNGKeyArray

from utils import make_mesh


class LatentMap(eqx.Module):
    """Parametrized function to convert point in 2D (pixel coordinate) into a latent."""

    positions: Int[Array, "n dim"]
    neighbor_map: Int[Array, "w h num_neighbors"]
    embeddings: Array

    def __init__(self, key, positions, image):
        positions = jnp.array(positions)
        coords = make_mesh(image.max_shape())

        def kd_neighbors_callback(positions, coords):
            t = time.time()
            # strip nans from positions
            mask = jnp.logical_not(jnp.any(jnp.isnan(positions), axis=-1))
            positions_stripped = positions[mask]
            tree = scipy.spatial.KDTree(positions_stripped)
            _, nearest_indices = tree.query(coords, k=4)
            logging.info(f"spent {time.time() - t} seconds in kd callback")
            return nearest_indices.astype(jnp.int32)

        shape = jnp.broadcast_shapes((image.max_shape()[0] * image.max_shape()[1], 4))
        dtype = jnp.result_type(jnp.int32)
        out_type = jax.ShapeDtypeStruct(shape, dtype)
        nearest_indices = jax.pure_callback(
            kd_neighbors_callback, out_type, positions, coords, vectorized=False
        )

        neighbor_map = -jnp.ones((*image.max_shape(), 4))
        neighbor_map = neighbor_map.at[coords[:, 0], coords[:, 1]].set(nearest_indices)

        self.neighbor_map = neighbor_map.astype(jnp.int32)
        self.positions = positions
        self.embeddings = jr.normal(key, (len(positions), 32)) * 1e-3

    @jax.named_scope("LatentMap.check")
    def check(self):
        new_embeddings = eqx.error_if(
            self.embeddings,
            jnp.any(jnp.isnan(self.embeddings)),
            "NaN in embeddings: should never be the case",
        )
        return eqx.tree_at(lambda s: s.embeddings, self, new_embeddings)

    @jax.named_scope("LatentMap.call")
    def __call__(self, position: Float[Array, "2"]):
        int_pos = jnp.floor(position).astype(jnp.int32)
        neighbors = self.neighbor_map[int_pos[0], int_pos[1]]
        neighbors = eqx.error_if(
            neighbors, neighbors.sum() < 0, "Undefined neighbors: index out of bounds."
        )
        distances = jax.lax.stop_gradient(
            jnp.linalg.norm(self.positions[neighbors] - int_pos, axis=1)
        )

        if True:  # this branch is "joost's method" - which does not make any sense to me
            latent = jnp.sum(distances[:, None] * self.embeddings[neighbors], axis=0)
        else:
            weights = 1.0 / (distances + 1e-6)
            latent = jnp.sum(weights[:, None] * self.embeddings[neighbors], axis=0) / jnp.sum(
                weights
            )
        return latent


"""
latent_map = LatentMap(jr.PRNGKey(0), [(1, 1), (3, 3), (1, 3), (3, 1), (0, 0)], (4, 4))
assert jnp.allclose(latent_map(jnp.array([0, 0], dtype=jnp.int32)), latent_map.embeddings[4])
assert jnp.allclose(latent_map(jnp.array([1, 1], dtype=jnp.int32)), latent_map.embeddings[0])
assert jnp.allclose(
    latent_map(jnp.array([2, 2], dtype=jnp.int32)),
    jnp.mean(latent_map.embeddings[:-1], axis=0),
)
del latent_map
"""


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, key: PRNGKeyArray):
        keys = jr.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))

    @jax.named_scope("mlp.check")
    def check(self):
        def _check_layer(layer):
            return eqx.error_if(
                layer,
                jnp.logical_or(jnp.any(jnp.isnan(layer.weight)), jnp.any(jnp.isnan(layer.bias))),
                "NaN in weight or biases of the mlp: should never be the case",
            )

        new_layers = [_check_layer(layer) for layer in self.layers]
        return eqx.tree_at(lambda s: s.layers, self, new_layers)

    @jax.named_scope("mlp.call")
    def __call__(self, x: Float[Array, "latent_dim"]) -> Float[Array, "..."]:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x


class CombinedModel(eqx.Module):
    """Glue + output normalization."""

    latent_map: LatentMap
    mlp: MLP
    mu: Array
    std: Array

    def __init__(self, image, latent_map, mlp):
        self.mu = jnp.nanmean(image, axis=(0, 1))
        self.std = jnp.nanstd(image, axis=(0, 1))
        self.latent_map = latent_map
        self.mlp = mlp

    @jax.named_scope("combined.call")
    def __call__(self, x: Int[Array, "2"]) -> Float[Array, "..."]:
        latent = self.latent_map(x)
        out = self.mlp(latent) * jax.lax.stop_gradient(self.std) + jax.lax.stop_gradient(self.mu)
        return jax.lax.cond(
            jnp.any(jnp.isnan(out)),
            lambda: jax.lax.stop_gradient(self.mu),
            lambda: jnp.clip(out, 0, 1),
        )

    def check(self):
        # checks that all the weights that should not be nan are not nan
        new_latent_map = self.latent_map.check()
        new_mlp = self.mlp.check()
        cls = self
        cls = eqx.tree_at(lambda s: s.latent_map, cls, new_latent_map)
        cls = eqx.tree_at(lambda s: s.mlp, cls, new_mlp)
        return cls
