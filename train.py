import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import optax
from absl import logging
from absl.flags import FLAGS
from jax.scipy.signal import convolve
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from config import InrTypeEnum, SampleEnum
from config import get_config as C  # noqa
from model import MLP, CombinedModel, LatentMap
from utils import Image, make_mesh


@jax.named_scope("get_latent_points")
def get_latent_points(key: PRNGKeyArray, image: Image, fraction: float | int):
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
    if fraction < 1:
        num_latents = (image.shape[0] * image.shape[1] * fraction).astype(jnp.int32)
    else:
        num_latents = jnp.array(fraction).astype(jnp.int32)
    probs = grad_magnitude.reshape(-1)
    error_bool = jnp.array(False)  # noqa
    error_bool = jnp.logical_or(error_bool, jnp.count_nonzero(probs) < num_latents)
    error_bool = jnp.logical_or(error_bool, limit_num_latents < num_latents)
    if FLAGS.on_error == "stop":
        probs = eqx.error_if(
            probs, jnp.count_nonzero(probs) < num_latents, "Not enough pixels to sample"
        )
        probs = eqx.error_if(
            probs, limit_num_latents < num_latents, "The number to sample is higher than the limit"
        )

    indices = jr.choice(key, grid, shape=(int(limit_num_latents),), p=probs, replace=False)
    indices = jnp.where(
        (jnp.arange(limit_num_latents) < num_latents)[:, None].repeat(2, 1), indices, -1
    )
    if FLAGS.on_error == "skip":
        return jax.lax.cond(error_bool, lambda: -jnp.ones_like(indices), lambda: indices).astype(
            jnp.int32
        )
    return indices.astype(jnp.int32)


def loss_and_grads(model, coords, pixels):
    no_nan_pixels = jnp.where(jnp.isnan(pixels), 0.0, pixels)
    prod_mask = jnp.where(jnp.isnan(pixels), 0, 1)
    num_good_pixels = jnp.sum(jnp.any(jnp.logical_not(jnp.isnan(pixels)), axis=-1))
    num_good_pixels = jax.lax.cond(num_good_pixels == 0, lambda: 1, lambda: num_good_pixels)

    def loss_fn(m: CombinedModel) -> Float[Array, ""]:
        preds = eqx.filter_vmap(m)(coords)
        return jnp.sum(((preds - no_nan_pixels) * prod_mask).ravel() ** 2) / num_good_pixels

    return eqx.filter_value_and_grad(loss_fn)(model)


@jax.named_scope("sample_pixels")
def sample_pixels(
    image: Image,
    key: PRNGKeyArray,
    fraction: float,
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
def train_image(
    image: Image, key: PRNGKeyArray, epochs_list: list[int], decoder: MLP | None = None
) -> tuple[CombinedModel]:
    """Trains an MLP model to represent a single image."""
    image_data = image.data
    w, h = image.shape  # 'channels' is separate cuz it must be static
    c = image.channels

    if C().inr_type == InrTypeEnum.STANDARD:
        mlp_dim = int(C().latent_dim)
    else:
        mlp_dim = int((C().latent_dim + 2) * C().num_neighbours)

    k1, k2, k3, k4 = jr.split(key, 4)
    mlp = (
        decoder.freeze()
        if decoder is not None
        else MLP(mlp_dim, [int(C().dec_layers[1] * mlp_dim)], c, k1)
    )
    if len(C().dec_layers) > 3:
        _msg = "The decoder must have at most 1 middle layer"
        raise ValueError(_msg)
    latent_points = get_latent_points(k2, image, fraction=C().num_latents)

    latent_map = LatentMap(k3, latent_points, image)
    model = CombinedModel(image_data, latent_map, mlp)
    model = model.check()

    optim = optax.adam(learning_rate=C().enc_optimiser[1]["lr"])
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    def scan_fn(carry, _):
        model, opt_state, local_key = carry
        sample_key, subkey = jr.split(local_key)
        if C().coord_subsampler[0] == SampleEnum.SUB:
            batch_coords, batch_pixels = sample_pixels(
                image, subkey, fraction=C().coord_subsampler[1]["num_samples"]
            )
        else:
            raise NotImplementedError("Only SUB is supported for now")

        loss, grads = loss_and_grads(model, batch_coords, batch_pixels)
        if decoder is not None:
            grads = eqx.error_if(
                grads,
                jnp.any(grads.mlp.layers[0].weight != 0),
                "Decoder is frozen, but grad is not zero!",
            )
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)

        return (model, opt_state, sample_key), loss

    carry = (model, opt_state, k4)
    models = []
    for epochs in epochs_list:
        carry, _ = eqx.filter_jit(jax.lax.scan)(scan_fn, carry, None, length=epochs)
        models.append(carry[0])

    logging.info("Decoder training done")

    return (*models,)


@jax.named_scope("train_decoder")
def train_decoder(datagen, num_samples, key, epochs):  # noqa
    logging.info("Starting to train decoder...")
    k1, key = jr.split(key)
    _test_image, _ = next(datagen)
    c = _test_image.channels

    if C().inr_type == InrTypeEnum.STANDARD:
        mlp_dim = int(C().latent_dim)
    else:
        mlp_dim = int((C().latent_dim + 2) * C().num_neighbours)
    mlp = MLP(mlp_dim, [int(C().dec_layers[1] * mlp_dim)], c, k1)
    if len(C().dec_layers) > 3:
        _msg = "The decoder must have at most 1 middle layer"
        raise ValueError(_msg)

    if FLAGS.use_cached_decoder:
        mlp = eqx.tree_deserialise_leaves(".cached_decoder.eqx", mlp)
        return mlp
    mlp_opt = optax.adam(learning_rate=C().dec_optimiser[1]["lr"])
    mlp_opt_state = mlp_opt.init(eqx.filter(mlp, eqx.is_inexact_array))

    optim = optax.adam(learning_rate=C().enc_optimiser[1]["lr"])

    def make_model(image, key):
        k2, k3, k4, key = jr.split(key, 4)

        latent_points = get_latent_points(k2, image, fraction=C().num_latents)

        latent_map = LatentMap(k3, latent_points, image)
        model = CombinedModel(image.data, latent_map, mlp)
        model = model.check()
        return model

    logging.info("Loading images...")
    images = []
    max_w = max_h = 0
    while len(images) < num_samples:
        images.append(next(datagen)[0])
        w, h = images[-1].max_shape()
        if w > 500 or h > 500:
            break
        max_w = max(w, max_w)
        max_h = max(h, max_h)

    logging.info("Done loading images.")
    for i in range(len(images)):
        images[i] = images[i].enlarge((max_w, max_h))
    breakpoint()
    g_image_soa = jax.tree.map(lambda *args: jnp.stack(args), *images, is_leaf=eqx.is_array)

    def init_model_and_opt(image, subkey):
        model = make_model(image, subkey)
        model = eqx.tree_at(lambda x: x.mlp, model, mlp)
        opt_state = eqx.filter_jit(optim.init)(
            eqx.filter(model, model.latent_map_only(eqx.is_inexact_array))
        )
        return model.check(), opt_state

    g_model_soa, g_opt_state_soa = jax.vmap(init_model_and_opt)(
        g_image_soa, jr.split(key, len(images))
    )

    def epoch_step(carry, _):
        # this one is a single epoch, should return the updated models opt_states, per-many-images
        mlp, mlp_opt_state, model_soa, opt_state_soa, key = carry

        def batch_step(carry_inner, model_image_opt):
            # this one is a single batch, as in, per-image epoch
            model, image, opt_state = model_image_opt
            model = eqx.tree_at(lambda m: m.mlp, model, mlp)  # override the mlp part
            accumulated_mlp_grads, key = carry_inner

            sample_key, subkey = jr.split(key)
            if C().coord_subsampler[0] == SampleEnum.SUB:
                batch_coords, batch_pixels = sample_pixels(
                    image, subkey, fraction=C().coord_subsampler[1]["num_samples"]
                )
            else:
                raise NotImplementedError("Only SUB is supported for now")
            loss, grads = loss_and_grads(model, batch_coords, batch_pixels)

            grads = eqx.error_if(grads, jnp.isnan(loss), "Nan loss encountered, critical")
            grads = eqx.error_if(
                grads, jnp.any(jnp.isnan(grads.mlp.layers[0].weight)), "Nan grad in mlp, critical"
            )
            grads = eqx.error_if(
                grads,
                jnp.any(jnp.isnan(grads.latent_map.embeddings)),
                "Nan grad in latent_map, critical",
            )

            _prev_mlp_param = model.mlp.layers[0].weight[0, 0]

            mlp_grads = eqx.filter(grads, grads.mlp_only(eqx.is_inexact_array))
            other_grads = eqx.filter(grads, grads.latent_map_only(eqx.is_inexact_array))

            accumulated_mlp_grads = jax.tree.map(
                lambda x, y: x + y,
                accumulated_mlp_grads,
                mlp_grads.mlp,
                is_leaf=eqx.is_inexact_array,
            )

            updates, opt_state = optim.update(
                other_grads,
                opt_state,
                eqx.filter(model, model.latent_map_only(eqx.is_inexact_array)),
            )

            model = eqx.apply_updates(model, updates)
            model = eqx.error_if(
                model, model.mlp.layers[0].weight[0, 0] != _prev_mlp_param, "oh no"
            )

            return (accumulated_mlp_grads, key), (model, opt_state)

        accumulated_mlp_grads = jax.tree.map(jnp.zeros_like, eqx.filter(mlp, eqx.is_inexact_array))
        initial_inner_carry = (accumulated_mlp_grads, key)
        (accumulated_mlp_grads, _), (model_soa, opt_state_soa) = jax.lax.scan(
            batch_step,
            initial_inner_carry,
            (model_soa, g_image_soa, opt_state_soa),
        )

        accumulated_mlp_grads = jax.tree.map(lambda x: x / num_samples, accumulated_mlp_grads)
        mlp_updates, mlp_opt_state = mlp_opt.update(accumulated_mlp_grads, mlp_opt_state, mlp)
        mlp = eqx.apply_updates(mlp, mlp_updates)

        key_final = jr.split(key, 2)[0]
        return (mlp, mlp_opt_state, model_soa, opt_state_soa, key_final), None

    logging.info(
        "Starting decoder training (this takes a while, because I don't parallelize it;"
        " for imagenette it is like 4s * num_samples, for mnist it is 0.15s * num_samples)"
    )

    carry = (mlp, mlp_opt_state, g_model_soa, g_opt_state_soa, key)
    (mlp, _, _, _, _), _ = eqx.filter_jit(jax.lax.scan)(epoch_step, carry, None, length=epochs)

    logging.info("Decoder training done")
    eqx.tree_serialise_leaves(".cached_decoder.eqx", mlp)

    return mlp
