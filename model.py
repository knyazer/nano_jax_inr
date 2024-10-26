import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from utils import make_mesh


class LatentMap(eqx.Module):
    """Parametrized function to convert point in 2D (pixel coordinate) into a latent."""

    positions: Int[Array, "n dim"]
    neighbor_map: Int[Array, "w h num_neighbors"]
    embeddings: Array

    def __init__(self, key, positions, shape=(28, 28)):
        positions = jnp.array(positions)

        def compute_nearest_index(coord):
            distances = jnp.linalg.norm(positions - coord, axis=1)
            nearest_indices = jnp.argsort(distances)[:4]
            return nearest_indices

        coords = make_mesh(shape)
        nearest_indices = jax.lax.map(compute_nearest_index, coords)

        neighbor_map = jnp.zeros((shape[0], shape[1], 4))
        neighbor_map = neighbor_map.at[coords[:, 0], coords[:, 1]].set(nearest_indices)

        self.neighbor_map = neighbor_map.astype(jnp.int16)
        self.positions = positions
        self.embeddings = jr.normal(key, (len(positions), 32)) * 1e-3

    def __call__(self, position: Float[Array, "2"]):
        int_pos = jnp.floor(position).astype(jnp.int32)
        # linear-ish interpolation of neighbors

        neighbors = self.neighbor_map[int_pos[0], int_pos[1]]
        distances = jnp.linalg.norm(self.positions[neighbors] - int_pos, axis=1)
        weights = jax.lax.stop_gradient(1.0 / (distances + 1e-6))

        latent = jnp.sum(weights[:, None] * self.embeddings[neighbors], axis=0) / jnp.sum(weights)
        return latent


latent_map = LatentMap(jr.PRNGKey(0), [(1, 1), (3, 3), (1, 3), (3, 1), (0, 0)], (4, 4))
assert jnp.allclose(latent_map(jnp.array([0, 0], dtype=jnp.int32)), latent_map.embeddings[4])
assert jnp.allclose(latent_map(jnp.array([1, 1], dtype=jnp.int32)), latent_map.embeddings[0])
assert jnp.allclose(
    latent_map(jnp.array([2, 2], dtype=jnp.int32)),
    jnp.mean(latent_map.embeddings[:-1], axis=0),
)
del latent_map


class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, key: PRNGKeyArray):
        keys = jr.split(key, len(hidden_dims) + 1)
        self.layers = [
            eqx.nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            for i in range(len(hidden_dims))
        ]
        self.layers.append(eqx.nn.Linear(hidden_dims[-1], out_dim, key=keys[-1]))

    def __call__(self, x: Float[Array, "latent_dim"]) -> Float[Array, "..."]:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x


class CombinedModel(eqx.Module):
    """Needed to simplify the logic, and in case you desire to have some glue."""

    latent_map: LatentMap
    mlp: MLP
    mu: Array
    std: Array

    def __init__(self, image, latent_map, mlp):
        self.mu = jnp.mean(image, axis=(0, 1))
        self.std = jnp.std(image, axis=(0, 1))
        self.latent_map = latent_map
        self.mlp = mlp

    def __call__(self, x: Int[Array, "2"]) -> Float[Array, "..."]:
        latent = self.latent_map(x)
        return jnp.clip(self.mlp(latent) * self.std + self.mu, 0, 1)
