import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.spatial
from absl import logging
from jaxtyping import Array, Float, Int, PRNGKeyArray

from config import DistFuncEnum, HarmonicsFuncEnum, InrTypeEnum
from config import get_config as C  # noqa
from utils import make_mesh


## Some distance transformations
def dists_identity(dists):
    return dists


def dists_to1(dists):
    dists = dists / (dists.sum(axis=-1) + 1e-8)
    return dists


def dists_normalise(dists):
    mean = dists.view(-1).mean()
    std = dists.view(-1).std()
    dists = (dists - mean) / std
    return dists


def dists_neurbf(dists):
    dists = dists / (dists.sum(axis=-1) + 1e-8)
    dists = 1 - dists
    return dists


class LatentMap(eqx.Module):
    """Parametrized function to convert point in 2D (pixel coordinate) into a latent."""

    positions: Int[Array, "n dim"]
    neighbor_map: Int[Array, "w h num_neighbors"]
    embeddings: Array
    harmonics: Float[Array, "32"]

    def __init__(self, key, positions, image):
        positions = jnp.array(positions)
        coords = make_mesh(image.max_shape())

        def kd_neighbors_callback(positions, coords):
            t = time.time()  # noqa
            # strip nans from positions
            mask = jnp.logical_not(jnp.any(jnp.isnan(positions), axis=-1))
            positions_stripped = positions[mask]
            tree = scipy.spatial.KDTree(positions_stripped)
            _, nearest_indices = tree.query(coords, k=C().num_neighbours)
            return nearest_indices.astype(jnp.int32)

        shape = jnp.broadcast_shapes(
            (image.max_shape()[0] * image.max_shape()[1], C().num_neighbours)
        )
        dtype = jnp.result_type(jnp.int32)
        out_type = jax.ShapeDtypeStruct(shape, dtype)
        nearest_indices = jax.pure_callback(kd_neighbors_callback, out_type, positions, coords)

        neighbor_map = -jnp.ones((*image.max_shape(), C().num_neighbours))
        neighbor_map = neighbor_map.at[coords[:, 0], coords[:, 1]].set(nearest_indices)

        self.neighbor_map = neighbor_map.astype(jnp.int32)
        self.positions = positions
        self.embeddings = jr.uniform(
            key,
            (len(positions), C().latent_dim),
            minval=-C().latent_init_distr[1]["a"],
            maxval=C().latent_init_distr[1]["b"],
        )

        harmonics_range = jnp.array(
            [C().harmonics_method[1]["start"], C().harmonics_method[1]["end"]]
        )
        harmonics = jnp.linspace(
            jnp.log2(harmonics_range[0]), jnp.log2(harmonics_range[1]), C().latent_dim
        )
        harmonics = jnp.exp2(harmonics)
        self.harmonics = harmonics

    @jax.named_scope("LatentMap.check")
    def check(self):
        new_embeddings = eqx.error_if(
            self.embeddings,
            jnp.any(jnp.isnan(self.embeddings)),
            "NaN in embeddings: should never be the case",
        )
        return eqx.tree_at(lambda s: s.embeddings, self, new_embeddings)

    def harm_function(self, x):
        if C().harmonics_function == HarmonicsFuncEnum.SIN:
            return jnp.sin(x)
        if C().harmonics_function == HarmonicsFuncEnum.COS:
            return jnp.cos(x)
        _msg = f"Unknown harmonics function: {C().harmonics_function}"
        raise ValueError(_msg)

    @jax.named_scope("LatentMap.call")
    def __call__(self, position: Float[Array, "2"]):
        int_pos = jnp.floor(position).astype(jnp.int32)
        neighbors = self.neighbor_map[int_pos[0], int_pos[1]]
        neighbors = eqx.error_if(
            neighbors, neighbors.sum() < 0, "Undefined neighbors: index out of bounds."
        )
        latents = self.embeddings[neighbors]
        deltas = self.positions[neighbors] - int_pos
        distances = jax.lax.stop_gradient(jnp.linalg.norm(deltas, axis=1))

        if C().inr_type == InrTypeEnum.STANDARD:
            if C().dist_function == DistFuncEnum.SUMTO1_NONLINEAR:
                dist_fn = dists_to1
            elif C().dist_function == DistFuncEnum.IDENTITY:
                dist_fn = lambda x: x  # noqa
            elif C().dist_function == DistFuncEnum.NEURBF:
                dist_fn = dists_neurbf
            else:
                _msg = f"Unknown distance function: {C().dist_function}"
                raise ValueError(_msg)
            distances = dist_fn(distances)

            harmonized = distances[:, None] @ self.harmonics[None, :]
            out = self.harm_function(harmonized) * latents
            return out.sum(axis=-2)
        if C().inr_type == InrTypeEnum.RELPOS:
            d = deltas.astype(jnp.float32)
            if C().relpos_normalise:
                d = (d - d.mean()) / (d.std() + 1e-6)
            out = jnp.concatenate([d.ravel(), latents.ravel()], axis=-1).ravel()
            logging.info(f"Using relative position, output shape is {out.shape}")
            return out
        _msg = f"Unknown inr type: {C().inr_type}"
        raise ValueError(_msg)


class MLP(eqx.Module):
    trainable: Array
    layers: list[eqx.nn.Linear]

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, key: PRNGKeyArray):
        keys = jr.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))
        self.trainable = jnp.array(True)  # noqa

    def freeze(self):
        return eqx.tree_at(lambda s: s.trainable, self, False)  # noqa

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
        layers = jax.lax.cond(
            self.trainable, lambda x: x, lambda x: jax.lax.stop_gradient(x), self.layers
        )
        for layer in layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = layers[-1](x)
        return x


class CombinedModel(eqx.Module):
    """Glue + output normalization."""

    latent_map: LatentMap
    mlp: MLP

    def __init__(self, _image, latent_map, mlp):
        self.latent_map = latent_map
        self.mlp = mlp

    @jax.named_scope("combined.call")
    def __call__(self, x: Int[Array, "2"]) -> Float[Array, "..."]:
        latent = self.latent_map(x)
        out = self.mlp(latent)
        return jax.lax.cond(
            jnp.any(jnp.isnan(out)),
            lambda: jnp.zeros_like(out),
            lambda: out,
        )

    def check(self):
        # checks that all the weights that should not be nan are not nan
        new_latent_map = self.latent_map.check()
        new_mlp = self.mlp.check()
        cls = self
        cls = eqx.tree_at(lambda s: s.latent_map, cls, new_latent_map)
        cls = eqx.tree_at(lambda s: s.mlp, cls, new_mlp)
        return cls

    def mlp_only(self, filter_fn):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: filter_fn(leaf) & ("mlp" in jax.tree_util.keystr(path)), self
        )

    def latent_map_only(self, filter_fn):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: filter_fn(leaf) & ("latent_map" in jax.tree_util.keystr(path)), self
        )
