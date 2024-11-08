import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import optax
from jax.scipy.signal import convolve
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from model import MLP, CombinedModel, LatentMap
from utils import Image, make_mesh


@jax.named_scope("get_latent_points")
def get_latent_points(key: PRNGKeyArray, image: Image, fraction: float = 0.05):
    w, h, c = image.data.shape

    # the following piece of code is somewhat weirdly written:
    # the reason is that we wish to match kornia behaviour closely
    # somehow it is very hard to achieve, but what i did at least matches moments :)
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

    padded_image = jnp.pad(image.data, ((1, 1), (1, 1), (0, 0)), mode="edge")
    grad_x = jax.vmap(lambda i: convolve(i, sobel_x, mode="valid"), in_axes=-1)(padded_image)
    grad_x_sq = jnp.sum(grad_x**2, axis=0)

    grad_y = jax.vmap(lambda i: convolve(i, sobel_y, mode="valid"), in_axes=-1)(padded_image)
    grad_y_sq = jnp.sum(grad_y**2, axis=0)

    grad_magnitude = jnp.nan_to_num(jnp.sqrt(grad_x_sq + grad_y_sq))

    x_grid, y_grid = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing="ij")
    grid = jnp.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)

    limit_num_latents = image.max_latents()
    num_latents = (image.shape[0] * image.shape[1] * fraction).astype(jnp.int32)
    probs = grad_magnitude.reshape(-1)
    probs = eqx.error_if(
        probs, jnp.count_nonzero(probs) < num_latents, "Not enough pixels to sample"
    )
    probs = eqx.error_if(
        probs, limit_num_latents < num_latents, "The number to sample is higher than the limit"
    )

    indices = jr.choice(key, grid, shape=(limit_num_latents,), p=probs, replace=False)
    indices = jnp.where(
        (jnp.arange(limit_num_latents) < num_latents)[:, None].repeat(2, 1), indices, -1
    )
    return indices.astype(jnp.int32)


@jax.named_scope("train_step")
def train_step(
    model: CombinedModel,
    optim: PyTree,
    opt_state: PyTree,
    coords: Float[Array, "*"],
    pixels: Float[Array, "*"],
) -> tuple:
    # replace nan pixels with m.mu
    no_nan_pixels = jnp.where(jnp.isnan(pixels), model.mu, pixels)
    prod_mask = jnp.where(jnp.isnan(pixels), 0, 1)
    num_good_pixels = jnp.sum(jnp.any(jnp.logical_not(jnp.isnan(pixels)), axis=-1))
    num_good_pixels = jax.lax.cond(num_good_pixels == 0, lambda: 1, lambda: num_good_pixels)

    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.sum(((preds - no_nan_pixels) * prod_mask).ravel() ** 2) / num_good_pixels

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss


@jax.named_scope("sample_pixels")
def sample_pixels(
    image: Image,
    key: PRNGKeyArray,
    fraction=0.25,
) -> tuple:
    w, h = image.shape
    W, H = image.max_shape()  # noqa
    indices = jr.choice(
        key, W * H, shape=(int(max(min(W * H * fraction, W * H), 1)),), replace=False
    )
    coords = make_mesh((W, H))[indices]
    pixels = image.data[coords[:, 0], coords[:, 1], :]
    return coords, pixels


@eqx.filter_jit
@jax.named_scope("train_image")
def train_image(image: Image, key: PRNGKeyArray, epochs: int = 1000) -> CombinedModel:
    """Trains an MLP model to represent a single image."""
    image_data = image.data
    w, h = image.shape  # 'channels' is separate cuz it must be static
    c = image.channels

    k1, k2, k3, k4 = jr.split(key, 4)

    mlp = MLP(32, [32], c, k1)
    latent_points = get_latent_points(k2, image, fraction=0.05)

    latent_map = LatentMap(k3, latent_points, image)
    model = CombinedModel(image_data, latent_map, mlp)
    model = model.check()

    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        batch_coords, batch_pixels = sample_pixels(image, subkey, fraction=0.25)
        model, opt_state, loss = train_step(model, optim, opt_state, batch_coords, batch_pixels)
        return (model, opt_state, sample_key), loss

    jax.debug.print("{}", model.mlp.layers[0].weight[..., 0])
    (model, opt_state, _), losses = jax.lax.scan(
        scan_fn, (model, opt_state, k4), None, length=epochs
    )
    jax.debug.print("{}", model.mlp.layers[0].weight[..., 0])

    return model
