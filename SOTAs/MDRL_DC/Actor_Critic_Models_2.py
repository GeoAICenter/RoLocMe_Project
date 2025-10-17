import equinox as eqx
import jax
import jax.numpy as jnp


class Actor(eqx.Module):
    list_convs: list
    list_linears: list
    pooling_list: list
    relu: jax.nn.relu
    softmax: jax.nn.softmax

    def __init__(
            self,
            observation_space_dims,
            action_space_dims,
            key,
    ):
        linear_key, *conv_list = jax.random.split(key, 1 + 2)
        _, *linear_key = jax.random.split(linear_key, 1 + 3)
        self.list_convs = [
            eqx.nn.Conv2d(
                in_channels=observation_space_dims,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                key=conv_list[0]
            ),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                key=conv_list[1]
            )
        ]
        self.pooling_list = [
            eqx.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                use_ceil=True,
            )
        ]
        self.list_linears = [
            eqx.nn.Linear(
                in_features=6 * 6 * 64 + 128,
                out_features=256,
                key=linear_key[0]
            ),
            eqx.nn.Linear(
                in_features=256,
                out_features=128,
                key=linear_key[1]
            ),
            eqx.nn.Linear(
                in_features=128,
                out_features=action_space_dims,
                key=linear_key[2]
            )
        ]
        self.relu = jax.nn.relu
        self.softmax = jax.nn.softmax

    def __call__(
            self,
            x,
            building_embedding,
    ):
        x = self.relu(self.list_convs[0](x))
        x = self.pooling_list[0](x)
        x = self.relu(self.list_convs[1](x))
        x = jnp.ravel(x)
        x = jnp.concatenate([x, building_embedding])
        x = self.relu(self.list_linears[0](x))
        x = self.relu(self.list_linears[1](x))
        x = self.list_linears[2](x)
        # x = self.softmax(x)
        return x


class Critic(eqx.Module):
    list_convs: list
    list_linears: list
    pooling_list: list
    relu: jax.nn.relu

    def __init__(
            self,
            observation_space_dims,
            key,
    ):
        linear_key, *conv_list = jax.random.split(key, 1 + 2)
        _, *linear_key = jax.random.split(linear_key, 1 + 3)
        self.list_convs = [
            eqx.nn.Conv2d(
                in_channels=observation_space_dims,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                key=conv_list[0]
            ),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                key=conv_list[1]
            )
        ]
        self.pooling_list = [
            eqx.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                use_ceil=True,
            )
        ]
        self.list_linears = [
            eqx.nn.Linear(
                in_features=6 * 6 * 64 + 128,
                out_features=256,
                key=linear_key[0]
            ),
            eqx.nn.Linear(
                in_features=256,
                out_features=128,
                key=linear_key[1]
            ),
            eqx.nn.Linear(
                in_features=128,
                out_features=1,
                key=linear_key[2]
            )
        ]
        self.relu = jax.nn.relu

    def __call__(
            self,
            x,
            building_embedding,
    ):
        x = self.relu(self.list_convs[0](x))
        x = self.pooling_list[0](x)
        x = self.relu(self.list_convs[1](x))
        x = jnp.ravel(x)
        x = jnp.concatenate([x, building_embedding])
        x = self.relu(self.list_linears[0](x))
        x = self.relu(self.list_linears[1](x))
        x = self.list_linears[2](x)
        return x
