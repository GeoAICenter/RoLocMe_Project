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
            building_embedding_train,
            building_embedding_val,
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
        self.building_embedding_train = jnp.array(building_embedding_train)
        self.building_embedding_val = jnp.array(building_embedding_val)

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
            inputs
    ):
        (
            all_robots_positions,
            full_propagation_map,
            building_map,
            visited_count_map,
        ) = carry
        current_robot_index = inputs
        current_robot_position = all_robots_positions.at[current_robot_index].get()

        def single_robot(
                carry_,
                input_
        ):
            carry_image, current_robot_index_, all_robots_positions_ = carry_
            current_checking_robot_index = input_
            return (
                jax.lax.cond(
                    jnp.equal(current_checking_robot_index, current_robot_index_).sum(),
                    lambda _: carry_image,
                    lambda _: carry_image.at[
                        all_robots_positions_.at[current_checking_robot_index].get().at[0].get(),
                        all_robots_positions_.at[current_checking_robot_index].get().at[1].get()
                    ].add(1.0),
                    None
                ),
                current_robot_index_,
                all_robots_positions_
            ), None

        (other_robots_positions, _, _), _ = jax.lax.scan(
            single_robot,
            (
                jnp.zeros((256, 256)),
                current_robot_index.sum(),
                all_robots_positions
            ),
            jnp.arange(self.n_agents),
        )

        windowed_location_map = crop(
            5,
            current_robot_position[0],
            current_robot_position[1],
            jnp.zeros((256, 256)).at[current_robot_position[0], current_robot_position[1]].add(1.0),
        )

        windowed_sampling_map = crop(
            5,
            current_robot_position[0],
            current_robot_position[1],
            full_propagation_map * jnp.where(visited_count_map != 0.0, 1.0, 0.0)
        )

        windowed_visited_count_map = crop(
            5,
            current_robot_position[0],
            current_robot_position[1],
            visited_count_map
        ) / jnp.max(visited_count_map)

        windowed_building_map = crop(
            5,
            current_robot_position[0],
            current_robot_position[1],
            building_map
        )

        reduced_location_map = jax.image.resize(
            (jnp.zeros((256, 256)).at[current_robot_position[0], current_robot_position[1]].add(1.0)) * 255.0,
            (11, 11),
            "lanczos3",
        ) / 255.0
        reduced_team_location_map = jax.image.resize(
            (other_robots_positions / jnp.max(other_robots_positions)) * 255.0,
            (11, 11),
            "lanczos3"
        ) / 255.0
        reduced_sampling_map = jax.image.resize(
            (full_propagation_map * jnp.where(visited_count_map != 0.0, 1.0, 0.0)) * 255.0,
            (11, 11),
            "lanczos3"
        ) / 255.0
        reduced_visited_count_map = jax.image.resize(
            (visited_count_map / jnp.max(visited_count_map)) * 255.0,
            (11, 11),
            "lanczos3"
        ) / 255.0
        return carry, jnp.stack(
            [
                windowed_location_map,
                windowed_visited_count_map,
                windowed_sampling_map,
                windowed_building_map,
                reduced_location_map,
                reduced_visited_count_map,
                reduced_sampling_map,
                reduced_team_location_map
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(
            self,
            full_propagation_map,
            building_map,
            visited_count_map,
            robot_positions,
    ):
        _, obss = jax.lax.scan(
            self._get_obs_per_robot,
            (
                robot_positions,
                full_propagation_map,
                building_map,
                visited_count_map,
            ),
            jnp.arange(self.n_agents),
        )
        return obss

    @partial(jax.jit, static_argnums=(0,))
    def _reset_helper(
            self,
            reset_key,
            full_propagation_maps,
            environment_building_maps,
            shortest_dist_maps,
            building_embeddings,
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
        building_embedding = building_embeddings[chosen_map_index].squeeze()

        # Choose a random agent location
        row_t, col_t = jnp.argmax(full_propagation_map) // 256.0, jnp.argmax(full_propagation_map) % 256.0

        indices = jnp.arange(start=0, stop=256 * 256, step=1)
        euclidean_dist = jnp.sqrt(
            (indices // 256.0 - row_t) ** 2 + (indices % 256.0 - col_t) ** 2
        ).reshape(256, 256) * (1.0 - building_map)

        available_deployment_positions = jnp.where(euclidean_dist > self.minimum_deployment_from_transmitter, 1.0, 0.0)
        illegal_positions_map = jnp.where(shortest_dist_map != 0, 1.0, 0.0)
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
        obs = self.get_obs(
            full_propagation_map,
            building_map,
            visited_positions_global,
            robots_initial_positions,
        )

        env_state = {
            "env_map_index": chosen_map_index.squeeze(),
            "robots_positions": jax.lax.stop_gradient(robots_initial_positions.squeeze()),
            "action_mask": jax.lax.stop_gradient(legal_actions_mask.squeeze()),
            "time_step": jax.lax.stop_gradient(time_step.squeeze()),
            "visited_positions": visited_positions_global.squeeze(),
            "dist_of_closest_robot_2_transmitter": dist_of_closest_robot_2_transmitter.squeeze(),
            "obs": obs.squeeze(),
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
            building_embeddings=self.building_embedding_train,
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
            building_embeddings=self.building_embedding_val,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(
            self,
            joint_actions_indices,
            prev_dist_of_closest_robot_2_transmitter,
            next_dist_of_closest_robot_2_transmitter,
    ):
        count_idle = jnp.where(joint_actions_indices == 8, 1.0, 0.0).sum()
        reward = jax.lax.cond(
            jnp.less_equal(next_dist_of_closest_robot_2_transmitter, self.distance_within_to_terminate).sum(),
            lambda _: -1 * (self.n_agents - count_idle) + 100,
            lambda _: -1 * (self.n_agents - count_idle),
            None
        )
        return jnp.ones((self.n_agents,)) * reward

    @partial(jax.jit, static_argnums=(0,))
    def get_terminated(
            self,
            robots_positions,
            shortest_dist_map,
            time_step
    ):
        is_terminate = jax.lax.cond(
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
        return (jnp.ones((self.n_agents,)) * is_terminate).astype(jnp.bool_)

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
        dpm, building, shortest_dist_map = (
                self.full_propagation_maps_train.at[current_env_state["env_map_index"]].get(),
                self.environment_building_maps_train.at[current_env_state["env_map_index"]].get(),
                self.shortest_distance_map_train.at[current_env_state["env_map_index"]].get(),
        )

        time_step = current_env_state["time_step"] + 1
        visited_positions = current_env_state["visited_positions"]
        actions = self.individual_actions.at[joint_actions_indices].get()
        actions_with_m = jnp.where(joint_actions_indices % 2 != 0, 5, 5)
        robots_next_positions = current_env_state["robots_positions"] + actions * actions_with_m[:, None]
        robots_next_rows, robots_next_cols = robots_next_positions[:, 0], robots_next_positions[:, 1]

        robots_next_positions_map = jnp.zeros(
            shape=(256, 256),
            dtype=jnp.float32
        ).at[
            robots_next_rows,
            robots_next_cols
        ].add(1.0)

        next_visited_positions_map = visited_positions.at[robots_next_rows, robots_next_cols].add(1.0)

        next_obs = self.get_obs(
            dpm,
            building,
            next_visited_positions_map,
            robots_next_positions,
        )

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

        newest_dist_of_closest_robot_2_transmitter = next_state_shortest_dist

        reward = self.get_reward(
            joint_actions_indices,
            current_env_state["dist_of_closest_robot_2_transmitter"],
            newest_dist_of_closest_robot_2_transmitter,
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
            "dist_of_closest_robot_2_transmitter": newest_dist_of_closest_robot_2_transmitter.squeeze(),
            "visited_positions": jax.lax.stop_gradient(next_visited_positions_map.squeeze()),
            "obs": jax.lax.stop_gradient(next_obs.squeeze()),
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
            is_terminate,
            success,
            time_steps,
            total_actions,
    ):
        def perform(total_actions_):
            dpm, building, shortest_dist_map = (
                    self.full_propagation_maps_val.at[current_env_state["env_map_index"]].get(),
                    self.environment_building_maps_val.at[current_env_state["env_map_index"]].get(),
                    self.shortest_distance_map_val.at[current_env_state["env_map_index"]].get(),
            )
            time_step = current_env_state["time_step"] + 1
            visited_positions = current_env_state["visited_positions"]
            actions = self.individual_actions.at[joint_actions_indices].get()
            total_actions_ += jnp.where(joint_actions_indices != 8, 1, 0).sum()

            actions_with_m = jnp.where(joint_actions_indices % 2 != 0, 5, 5)
            robots_next_positions = current_env_state["robots_positions"] + actions * actions_with_m[:, None]
            robots_next_rows, robots_next_cols = robots_next_positions[:, 0], robots_next_positions[:, 1]

            next_visited_positions_map = visited_positions.at[robots_next_rows, robots_next_cols].add(1.0)

            next_obs = self.get_obs(
                dpm,
                building,
                next_visited_positions_map,
                robots_next_positions,
            )

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

            newest_dist_of_closest_robot_2_transmitter = next_state_shortest_dist

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
                "obs": jax.lax.stop_gradient(next_obs.squeeze()),
            }
            success_ = jnp.less_equal(newest_dist_of_closest_robot_2_transmitter, self.distance_within_to_terminate).sum()
            return next_env_state, jnp.all(terminated), success_, time_step, total_actions_

        return jax.lax.cond(
            is_terminate,
            lambda _: (current_env_state, is_terminate, success, time_steps, total_actions),
            lambda _: perform(total_actions),
            None
        )
