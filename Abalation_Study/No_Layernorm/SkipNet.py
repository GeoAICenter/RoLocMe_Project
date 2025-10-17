import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import optax
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import numpy as np
import random

class Encoder(eqx.Module):
    conv_layers: list
    average_pooling_layers: list
    leaky_relu_alpha: float
    leaky_relu: jax.nn.leaky_relu

    def __init__(
            self,
            enc_in: int,
            enc_out: int,
            n_dim: int,
            leaky_relu_alpha: float,
            key: jax.Array
    ):
        new_key, *conv_keys = jax.random.split(
            key,
            1 + 12
        )

        self.conv_layers = [
                               eqx.nn.Conv2d(
                                   in_channels=enc_in,
                                   out_channels=n_dim,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=1,
                                   key=conv_keys[0]
                               )
                           ] + [
                               eqx.nn.Conv2d(
                                   in_channels=n_dim,
                                   out_channels=n_dim,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=1,
                                   key=conv_keys[i + 1]
                               ) for i in range(8)
                           ]

        self.conv_layers.append(
            eqx.nn.Conv2d(
                in_channels=n_dim,
                out_channels=4,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                key=conv_keys[-1]
            )
        )

        self.average_pooling_layers = [
            eqx.nn.AvgPool2d(
                kernel_size=(2, 2),
                stride=2
            )
            for _ in range(3)
        ]

        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_relu = jax.nn.leaky_relu
        _ = self._init_weights(key=new_key)

    def _init_weights(self, key):
        new_key, *weight_initializer_key = jax.random.split(key, 11)
        where = lambda l: l.weight
        for i, layer in enumerate(self.conv_layers[:-3]):
            new_weights = jax.nn.initializers.he_normal()(
                weight_initializer_key[i],
                layer.weight.shape,
                dtype=jnp.float32
            )

            self.conv_layers[i] = eqx.tree_at(
                where,
                layer,
                new_weights
            )
        return new_key

    def __call__(self, x: Float[Array, "2 256 256"]):
        x = self.leaky_relu(self.conv_layers[0](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[1](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[2](x), self.leaky_relu_alpha)
        skip_1 = x
        x = self.average_pooling_layers[0](x)

        x = self.leaky_relu(self.conv_layers[3](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[4](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[5](x), self.leaky_relu_alpha)
        skip_2 = x
        x = self.average_pooling_layers[1](x)

        x = self.leaky_relu(self.conv_layers[6](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[7](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_layers[8](x), self.leaky_relu_alpha)
        skip_3 = x
        x = self.average_pooling_layers[2](x)
        x = self.leaky_relu(self.conv_layers[9](x), self.leaky_relu_alpha)
        return x, skip_1, skip_2, skip_3


class Decoder(eqx.Module):
    conv_transpose_layers: list
    leaky_relu_alpha: float
    leaky_relu: jax.nn.leaky_relu
    conv2d_output: eqx.nn.Conv2d

    def __init__(self,
                 dec_in: int,
                 dec_out: int,
                 n_dim: int,
                 leaky_relu_alpha: float,
                 key: jax.Array):
        new_key, conv_key, *conv_tranpose_keys = jax.random.split(
            key,
            12
        )

        self.conv_transpose_layers = [
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=dec_in,
                                             out_channels=dec_in,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[0]
                                         ),
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=dec_in + n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[1]
                                         )
                                     ] + [
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[i + 1]) if i % 3 != 2
                                         else eqx.nn.ConvTranspose2d(
                                             in_channels=2 * n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[i + 1]
                                         )
                                         for i in range(8)
                                     ]

        self.conv2d_output = eqx.nn.Conv2d(
            n_dim,
            dec_out,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            key=conv_key
        )
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_relu = jax.nn.leaky_relu
        _ = self._init_weights(key=new_key)

    def _init_weights(self, key: jax.Array):
        new_key, conv_weight_initializer_key, *weight_initializer_key = jax.random.split(key, 12)
        where = lambda l: l.weight
        for i, layer in enumerate(self.conv_transpose_layers):
            new_weights = jax.nn.initializers.he_normal()(
                weight_initializer_key[i],
                layer.weight.shape,
                dtype=jnp.float32
            )

            self.conv_transpose_layers[i] = eqx.tree_at(
                where,
                layer,
                new_weights
            )

        new_conv_weights = jax.nn.initializers.he_normal()(
            conv_weight_initializer_key,
            self.conv2d_output.weight.shape,
            dtype=jnp.float32
        )
        self.conv2d_output = eqx.tree_at(where,
                                         self.conv2d_output,
                                         new_conv_weights)
        return new_key

    def __call__(self,
                 x: Float[Array, "27 32 32"],
                 skip_1: Float[Array, "27 256 256"],
                 skip_2: Float[Array, "27 128 128"],
                 skip_3: Float[Array, "27 64 64"]):
        x = self.leaky_relu(self.conv_transpose_layers[0](x), self.leaky_relu_alpha)
        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_3], axis=0)

        x = self.leaky_relu(self.conv_transpose_layers[1](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[2](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[3](x), self.leaky_relu_alpha)

        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_2], axis=0)

        x = self.leaky_relu(self.conv_transpose_layers[4](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[5](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[6](x), self.leaky_relu_alpha)

        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_1], axis=0)

        x = self.leaky_relu(self.conv_transpose_layers[7](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[8](x), self.leaky_relu_alpha)
        x = self.leaky_relu(self.conv_transpose_layers[9](x), self.leaky_relu_alpha)

        x = self.conv2d_output(x)
        return x


class LocNet(eqx.Module):
    Encoder_model: Encoder
    Decoder_model: Decoder

    def __init__(self, enc_in, enc_out, dec_in, dec_out, enc_key, dec_key, n_dim, leaky_relu_alpha=0.3):
        self.Encoder_model = Encoder(
            enc_in=enc_in,
            enc_out=enc_out,
            n_dim=n_dim,
            leaky_relu_alpha=leaky_relu_alpha,
            key=enc_key
        )

        self.Decoder_model = Decoder(
            dec_in=dec_in,
            dec_out=dec_out,
            n_dim=n_dim,
            leaky_relu_alpha=leaky_relu_alpha,
            key=dec_key
        )

    def __call__(
            self,
            x: Float[Array, "2 256 256"]
    ):
        x, skip_1, skip_2, skip_3 = self.Encoder_model(x)
        x = self.Decoder_model(x, skip_1, skip_2, skip_3)
        return x


if __name__ == "__main__":
    pass
