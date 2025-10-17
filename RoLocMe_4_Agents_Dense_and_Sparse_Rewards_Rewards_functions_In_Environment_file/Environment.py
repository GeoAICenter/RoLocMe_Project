from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Float
from typing import Tuple
from utils import *


class Environment:
    def __init__(
            self,
            max_time_step,
            n_agents,
            minimum_distance_deployment_from_transmitter,
            m_per_step,
            full_propagation_maps_train,
            shortest_distance_map_train,
            shortest_distance_map_val,
            full_propagation_maps_val,
            building_maps_train,
            building_maps_val,
            distance_within_to_terminate,
    ):
        # Initialize number of agents
        self.distance_within_to_terminate = distance_within_to_terminate
        self.n_agents = n_agents
        self.m_per_step = m_per_step
        self.max_time_step = max_time_step
        self.minimum_deployment_from_transmitter = minimum_distance_deployment_from_transmitter
        self.individual_actions = jnp.array(
            [
                [1, 0],  # D
                [1, 1],  # DR
                [0, 1],  # R
                [-1, 1],  # RU
                [-1, 0],  # U
                [-1, -1],  # LU
                [0, -1],  # L
                [1, -1],  # LD
                [0, 0],  # Nothing
            ]
        )

        # Initialize all necessary environment maps to the memory.
        self.full_propagation_maps_train = jnp.array(full_propagation_maps_train)
        self.full_propagation_maps_val = jnp.array(full_propagation_maps_val)
        self.environment_building_maps_train = jnp.array(building_maps_train)
        self.environment_building_maps_val = jnp.array(building_maps_val)
        self.shortest_distance_map_train = jnp.array(shortest_distance_map_train)
        self.shortest_distance_map_val = jnp.array(shortest_distance_map_val)

    @partial(jax.jit, static_argnums=(0,))
    def _is_valid_move(
            self,
            carry,
            action_n0,
    ):
        (
            building_map,
            robot_position,
        ) = carry

        def check_move():
            iter_row_col = jnp.arange(
                1,
                self.m_per_step + 1
            ) * jnp.expand_dims(
                self.individual_actions.at[action_n0].get(),
                axis=0
            ).T + jnp.expand_dims(
                robot_position,
                axis=0
            ).T

            return jnp.logical_not(
                jax.lax.cond(
                    jnp.any(iter_row_col.flatten() < 0) | jnp.any(iter_row_col.flatten() >= 256),
                    lambda _: True,
                    lambda _: jax.lax.cond(
                        building_map.at[iter_row_col[0], iter_row_col[1]].get().sum() != 0.0,
                        lambda _: True,
                        lambda _: False,
                        None
                    ),
                    None
                )
            )

        return carry, check_move()

    @partial(jax.jit, static_argnums=(0,))
    def get_legal_mask(
            self,
            carry,
            input_,
    ):
        building_map = carry
        robots_position = input_
        _, masks = jax.lax.scan(
            self._is_valid_move,
            (
                building_map,
                robots_position,
            ),
            jnp.arange(9)
        )
        return carry, jnp.ravel(masks)

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs_per_robot(
            self,
            carry,
            robot_position
    ):
        (
            full_propagation_map,
            building_map,
            visited_count_map,
            locnet_pred,
        ) = carry
        locnet_pred = (locnet_pred - jnp.min(locnet_pred)) / (jnp.max(locnet_pred) - jnp.min(locnet_pred))
        windowed_sampling_map = crop(
            5,
            robot_position[0],
            robot_position[1],
            full_propagation_map * jnp.where(visited_count_map != 0, 1.0, 0.0)
        )
        building_map_ = (0.0 - building_map)
        windowed_building_map = crop(
            5,
            robot_position[0],
            robot_position[1],
            building_map_.at[robot_position[0], robot_position[1]].set(1.0)
        )
        windowed_pred_prop_map = crop(
            5,
            robot_position[0],
            robot_position[1],
            locnet_pred,
        )

        reduced_location_map = jax.image.resize(
            building_map_.at[robot_position[0], robot_position[1]].set(1.0),
            (11, 11),
            "bilinear",
        )
        reduced_sampling_map = jax.image.resize(
            full_propagation_map * jnp.where(visited_count_map != 0, 1.0, 0.0),
            (11, 11),
            "bilinear"
        )
        reduced_pred_locnet_map = jax.image.resize(
            locnet_pred,
            (11, 11),
            "bilinear",
        )

        return carry, jnp.stack(
            [
                windowed_sampling_map,
                reduced_sampling_map,
                windowed_building_map,
                reduced_location_map,
                windowed_pred_prop_map,
                reduced_pred_locnet_map,
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
            self,
            robot_positions,
            full_propagation_map,
            building_map,
            visited_count_map,
            locnet_pred_map,
    ):
        _, obss = jax.lax.scan(
            self._get_obs_per_robot,
            (
                full_propagation_map,
                building_map,
                visited_count_map,
                locnet_pred_map,
            ),
            robot_positions,
        )
        return obss

    @partial(jax.jit, static_argnums=(0,))
    def _reset_helper(
            self,
            reset_key,
            full_propagation_maps,
            environment_building_maps,
            shortest_dist_maps,
    ):
        (
            new_reset_key,
            environment_random_key,
            agents_random_keys
        ) = jax.random.split(
            reset_key,
            1 + 1 + 1
        )

        # Chose a random environment
        chosen_map_index = jax.random.randint(
            key=environment_random_key,
            minval=0,
            maxval=full_propagation_maps.shape[0],
            shape=(1,)
        )

        # Load the environment from the memory
        full_propagation_map = full_propagation_maps.at[chosen_map_index].get().squeeze()
        building_map = environment_building_maps.at[chosen_map_index].get().squeeze()
        shortest_dist_map = shortest_dist_maps.at[chosen_map_index].get().squeeze()

        # Choose a random agent location
        row_t, col_t = jnp.argmax(full_propagation_map) // 256.0, jnp.argmax(full_propagation_map) % 256.0

        indices = jnp.arange(start=0, stop=256 * 256, step=1)
        euclidean_dist = jnp.sqrt(
            (indices // 256.0 - row_t) ** 2 + (indices % 256.0 - col_t) ** 2
        ).reshape(256, 256) * (1.0 - building_map)

        available_deployment_positions = jnp.where(euclidean_dist > self.minimum_deployment_from_transmitter, 1.0, 0.0)
        illegal_positions_map = jnp.where((shortest_dist_map != 0) & (shortest_dist_map != -1), 1.0, 0.0)
        probability = 1.0 / jnp.count_nonzero(available_deployment_positions * illegal_positions_map).sum()

        robots_initial_indices = jax.random.choice(
            key=agents_random_keys,
            a=indices,
            p=(available_deployment_positions * illegal_positions_map).flatten() * probability,
            shape=(self.n_agents,)
        )

        robots_initial_row = robots_initial_indices // 256
        robots_initial_col = robots_initial_indices % 256
        robots_initial_positions = jnp.stack([robots_initial_row, robots_initial_col]).transpose()

        # Initialize visited position matrix
        visited_positions_global = jnp.zeros(shape=(256, 256), dtype=jnp.float32)
        visited_positions_global = visited_positions_global.at[robots_initial_row, robots_initial_col].set(1.0)

        time_step = jnp.zeros(shape=(1,), dtype=jnp.int32)

        _, legal_actions_mask = jax.lax.scan(
            self.get_legal_mask,
            building_map,
            robots_initial_positions,
        )

        dist_of_closest_robot_2_transmitter = jnp.min(
            shortest_dist_map.at[robots_initial_row, robots_initial_col].get()
        )

        env_state = {
            "env_map_index": chosen_map_index.squeeze(),
            "robots_positions": jax.lax.stop_gradient(robots_initial_positions.squeeze()),
            "action_mask": jax.lax.stop_gradient(legal_actions_mask.squeeze()),
            "time_step": jax.lax.stop_gradient(time_step.squeeze()),
            "visited_positions": visited_positions_global.squeeze(),
            "dist_of_closest_robot_2_transmitter": dist_of_closest_robot_2_transmitter.squeeze(),
        }
        return new_reset_key, env_state

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self,
            reset_key,
    ):
        return self._reset_helper(
            reset_key=reset_key,
            full_propagation_maps=self.full_propagation_maps_train,
            environment_building_maps=self.environment_building_maps_train,
            shortest_dist_maps=self.shortest_distance_map_train,
        )

    @partial(jax.jit, static_argnums=(0,))
    def validation_env_reset(
            self,
            val_reset_key
    ):
        return self._reset_helper(
            reset_key=val_reset_key,
            full_propagation_maps=self.full_propagation_maps_val,
            environment_building_maps=self.environment_building_maps_val,
            shortest_dist_maps=self.shortest_distance_map_val,
        )
    # Sparse rewards
    # @partial(jax.jit, static_argnums=(0,))
    # def get_reward(
    #        self,
    #        joint_actions_indices,
    #        prev_dist_of_closest_robot_2_transmitter,
    #        next_dist_of_closest_robot_2_transmitter,
    # ):
    #    count_idle = jnp.where(joint_actions_indices == 8, 1.0, 0.0).sum()
    #    return jax.lax.cond(
    #        jnp.less_equal(next_dist_of_closest_robot_2_transmitter, self.distance_within_to_terminate).sum(),
    #        lambda _: -1 * (self.n_agents - count_idle) + 100.0,
    #        lambda _: -1 * (self.n_agents - count_idle),
    #        None
    #    )
    # Dense rewards
    @partial(jax.jit, static_argnums=(0,))
    def get_reward(
            self,
            joint_actions_indices,
            prev_dist_of_closest_robot_2_transmitter,
            next_dist_of_closest_robot_2_transmitter,
    ):
        count_idle = jnp.where(joint_actions_indices == 8, 1.0, 0.0).sum()
        return jax.lax.cond(
            jnp.less(next_dist_of_closest_robot_2_transmitter, prev_dist_of_closest_robot_2_transmitter).sum(),
            lambda _: -1 * (self.n_agents - count_idle) + 1,
            lambda _: -1 * (self.n_agents - count_idle) - 1,
            None
        )

    # @partial(jax.jit, static_argnums=(0,))
    # def get_reward(
    #         self,
    #         prev_dist_of_closest_robot_2_transmitter,
    #         next_dist_of_closest_robot_2_transmitter,
    # ):
    #     reward = jax.lax.cond(
    #         jnp.less(next_dist_of_closest_robot_2_transmitter, prev_dist_of_closest_robot_2_transmitter).sum(),
    #         lambda _: 4,
    #         lambda _: -1,
    #         None
    #     )
    #     reward = jax.lax.cond(
    #         jnp.less_equal(next_dist_of_closest_robot_2_transmitter, self.distance_within_to_terminate).sum(),
    #         lambda _: 100,
    #         lambda _: reward,
    #         None
    #     )
    #     return reward

    @partial(jax.jit, static_argnums=(0,))
    def get_terminated(
            self,
            robots_positions,
            shortest_dist_map,
            time_step
    ):
        return jax.lax.cond(
            (
                    time_step >= self.max_time_step
            ) |
            (
                jnp.less_equal(
                    jnp.min(shortest_dist_map.at[robots_positions[:, 0], robots_positions[:, 1]].get()),
                    self.distance_within_to_terminate,
                )
            ),
            lambda _: True,
            lambda _: False,
            None
        )

    @partial(jax.jit, static_argnums=(0,))
    def continuous_reset(
            self,
            reset_env_key,
            state_,
            terminated,
    ):
        new_reset_env_key, new_state_ = self.reset(reset_env_key)
        (reset_env_key, state_) = jax.lax.cond(
            terminated.sum(),
            lambda _: (new_reset_env_key, new_state_),
            lambda _: (reset_env_key, state_),
            None
        )
        return reset_env_key, state_

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            joint_actions_indices,
            current_env_state,
            environment_reset_key,
    ):
        shortest_dist_map = self.shortest_distance_map_train.at[current_env_state["env_map_index"]].get()
        building = self.environment_building_maps_train.at[current_env_state["env_map_index"]].get()

        time_step = current_env_state["time_step"] + 1
        visited_positions = current_env_state["visited_positions"]
        actions = self.individual_actions.at[joint_actions_indices].get()

        robots_next_positions = current_env_state["robots_positions"] + actions * self.m_per_step
        robots_next_rows, robots_next_cols = robots_next_positions[:, 0], robots_next_positions[:, 1]
        next_visited_positions_map = visited_positions.at[robots_next_rows, robots_next_cols].add(1.0)

        _, next_valid_actions_mask = jax.lax.scan(
            self.get_legal_mask,
            building,
            robots_next_positions
        )
        next_valid_actions_mask = jax.lax.cond(
            jnp.array_equal(robots_next_positions, current_env_state["robots_positions"]).sum(),
            lambda _: next_valid_actions_mask.at[:, -1].set(False),
            lambda _: next_valid_actions_mask,
            None,
        )

        next_state_shortest_dist = jnp.min(
            shortest_dist_map.at[robots_next_rows, robots_next_cols].get()
        )

        reward = self.get_reward(
            joint_actions_indices,
            current_env_state["dist_of_closest_robot_2_transmitter"],
            next_state_shortest_dist,
        )

        newest_dist_of_closest_robot_2_transmitter = next_state_shortest_dist

        # reward = self.get_reward(
        #     current_env_state["dist_of_closest_robot_2_transmitter"],
        #     next_state_shortest_dist,
        # )
        # newest_dist_of_closest_robot_2_transmitter = jnp.minimum(
        #     next_state_shortest_dist,
        #     current_env_state["dist_of_closest_robot_2_transmitter"],
        # )

        terminated = self.get_terminated(
            robots_next_positions,
            shortest_dist_map,
            time_step,
        )

        next_env_state = {
            "env_map_index": current_env_state["env_map_index"],
            "robots_positions": jax.lax.stop_gradient(robots_next_positions.squeeze()),
            "action_mask": jax.lax.stop_gradient(next_valid_actions_mask.squeeze()),
            "time_step": jax.lax.stop_gradient(time_step.squeeze()),
            "dist_of_closest_robot_2_transmitter": newest_dist_of_closest_robot_2_transmitter.squeeze(),
            "visited_positions": jax.lax.stop_gradient(next_visited_positions_map.squeeze()),
        }

        environment_reset_key, next_env_state = self.continuous_reset(
            environment_reset_key,
            next_env_state,
            terminated,
        )

        return environment_reset_key, next_env_state, reward, terminated

    @partial(jax.jit, static_argnums=(0,))
    def val_step(
            self,
            joint_actions_indices,
            current_env_state,
            terminate_,
            success,
            time_takes,
            action_takes,
    ):

        def perform(action_takes_):
            building = self.environment_building_maps_val.at[current_env_state["env_map_index"]].get()
            shortest_dist_map = self.shortest_distance_map_val.at[current_env_state["env_map_index"]].get()
            time_step = current_env_state["time_step"] + 1
            visited_positions = current_env_state["visited_positions"]
            actions = self.individual_actions.at[joint_actions_indices].get()
            action_takes_ += jnp.where(joint_actions_indices != 8, 1, 0).sum()
            robots_next_positions = current_env_state["robots_positions"] + actions * self.m_per_step
            robots_next_rows, robots_next_cols = robots_next_positions[:, 0], robots_next_positions[:, 1]

            next_visited_positions_map = visited_positions.at[robots_next_rows, robots_next_cols].add(1.0)

            _, next_valid_actions_mask = jax.lax.scan(
                self.get_legal_mask,
                building,
                robots_next_positions
            )

            next_valid_actions_mask = jax.lax.cond(
                jnp.array_equal(robots_next_positions, current_env_state["robots_positions"]).sum(),
                lambda _: next_valid_actions_mask.at[:, -1].set(False),
                lambda _: next_valid_actions_mask,
                None,
            )

            terminated = self.get_terminated(
                robots_next_positions,
                shortest_dist_map,
                time_step,
            )

            next_env_state = {
                "env_map_index": current_env_state["env_map_index"],
                "robots_positions": jax.lax.stop_gradient(robots_next_positions.squeeze()),
                "action_mask": jax.lax.stop_gradient(next_valid_actions_mask.squeeze()),
                "time_step": jax.lax.stop_gradient(time_step.squeeze()),
                "visited_positions": jax.lax.stop_gradient(next_visited_positions_map.squeeze()),
                "dist_of_closest_robot_2_transmitter": current_env_state["dist_of_closest_robot_2_transmitter"].squeeze(),
            }
            success_ = jnp.less_equal(
                jnp.min(shortest_dist_map.at[robots_next_positions[:, 0], robots_next_positions[:, 1]].get()),
                self.distance_within_to_terminate
            )

            return next_env_state, terminated, success_, time_step, action_takes_

        return jax.lax.cond(
            terminate_.sum(),
            lambda _: (current_env_state, terminate_, success, time_takes, action_takes),
            lambda _: perform(action_takes),
            None
        )
