import equinox as eqx
import jax
import jax.numpy as jnp


class DuelingDQN(eqx.Module):
    list_convs: list
    A_list: list
    Ax_normalize: list
    Vx_normalize: list
    relu: jax.nn.relu

    def __init__(
            self,
            observation_space_dims,
            action_space_dims,
            key,
    ):
        linear_key, *conv_list = jax.random.split(key, 1 + 3)
        _, *linear_key = jax.random.split(linear_key, 1 + 4)
        self.list_convs = [
            eqx.nn.Conv2d(
                in_channels=observation_space_dims,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                key=conv_list[0]
            ),
            eqx.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                use_ceil=True,
            ),
            eqx.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                key=conv_list[2]
            )
        ]

        self.A_list = [
            eqx.nn.Linear(
                in_features=6 * 6 * 64,
                out_features=128,
                key=linear_key[2]
            ),
            eqx.nn.Linear(
                in_features=128,
                out_features=action_space_dims,
                key=linear_key[3]
            )
        ]

        self.Vx_normalize = [
            eqx.nn.LayerNorm((11, 11)),
            eqx.nn.LayerNorm((6, 6)),
        ]
        self.Ax_normalize = [
            eqx.nn.LayerNorm(128),
        ]

        self.relu = jax.nn.relu

    def __call__(
            self,
            x,
    ):
        x = self.relu(jax.vmap(self.Vx_normalize[0])(self.list_convs[0](x)))
        x = self.list_convs[1](x)
        x = self.relu(jax.vmap(self.Vx_normalize[1])(self.list_convs[2](x)))
        ft_extr = jnp.ravel(x)

        # A branch
        Ax = self.relu(self.Ax_normalize[0](self.A_list[0](ft_extr)))
        Q_values = self.A_list[1](Ax)
        return Q_values
