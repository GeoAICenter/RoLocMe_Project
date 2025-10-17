# All jax libraries
import cv2
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import jax
import jax.numpy as jnp
import keras
import equinox as eqx
import optax
from jax_tqdm import scan_tqdm
from functools import partial
from Environment_2 import Environment
from Dueling_DQN import DuelingDQN
from utils import get_all_maps
import numpy as np
from SkipNet import LocNet
from torch.utils.tensorboard import SummaryWriter
from CONSTANTS import *


@jax.jit
def extracted_maps(
        array,
        index
):
    return array.at[index].get()


def picking_random_action(
        legal_mask_,
        random_action_key_,
):
    return jax.random.choice(
        key=random_action_key_,
        a=jnp.arange(9),
        p=legal_mask_ * (1.0 / jnp.count_nonzero(legal_mask_)),
        replace=False,
        shape=(),
    )


@jax.jit
def select_actions_n_total_qvalue(
        model_dynamic_params,
        model_static_params,
        obs,
        legal_actions,
):
    duel_dqn_model_ = eqx.combine(model_dynamic_params, model_static_params)
    duel_dqn_q_values = jax.vmap(duel_dqn_model_)(obs)
    duel_dqn_q_values = jnp.where(legal_actions == 0, -1e6, duel_dqn_q_values)
    joint_actions = jnp.argmax(duel_dqn_q_values, axis=1)
    return joint_actions, jnp.max(duel_dqn_q_values, axis=1).sum()


@jax.jit
def get_target(inputs):
    q_values, rewards, terminated = inputs
    q_values = jnp.transpose(q_values, (1, 0, 2, 3))
    rewards = jnp.transpose(rewards, (1, 0))
    terminated = jnp.transpose(terminated, (1, 0))

    last_q = jnp.max(q_values.at[-1].get(), axis=-1)
    last_q = jnp.sum(last_q, axis=-1)
    lambda_return = rewards.at[-2].get() + GAMMA * (1.0 - terminated.at[-2].get()) * last_q

    last_q = jnp.max(q_values.at[-2].get(), axis=-1)
    last_q = jnp.sum(last_q, axis=-1)

    def calc_n_step_rewards(lambda_returns_and_next_q, transition_per_step):
        lambda_returns, next_q_value = lambda_returns_and_next_q
        done, q_val, reward = transition_per_step
        target_bootstrap = reward + GAMMA * (1 - done) * next_q_value
        delta = lambda_returns - next_q_value
        lambda_returns = target_bootstrap + GAMMA * GAMMA_LAMBDA * delta
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = jnp.sum(jnp.max(q_val, axis=-1), axis=-1)
        return (lambda_returns, next_q), lambda_returns

    _, targets = jax.lax.scan(
        calc_n_step_rewards,
        (lambda_return, last_q),
        (terminated.at[:-2].get(), q_values.at[:-2].get(), rewards.at[:-2].get()),
        reverse=True
    )
    lambda_targets = jnp.concatenate((targets, lambda_return[jnp.newaxis]))
    lambda_targets = jnp.transpose(lambda_targets, (1, 0))
    return lambda_targets


@jax.jit
def greedy_eps_selection(
        duel_dqn_params_,
        obs,
        legal_actions_,
        greedy_eps_key,
        eps_rate_,
):
    eps_key, *eps_choosing_keys = jax.random.split(greedy_eps_key, 1 + NUM_AGENTS)
    random_act_prob = jax.random.uniform(eps_key, minval=0, maxval=1)
    eps_choosing_keys = jnp.array(eps_choosing_keys)
    # (n_agent,)
    random_joint_actions = jax.vmap(picking_random_action)(
        legal_actions_,
        eps_choosing_keys
    )

    duel_dqn_model_ = eqx.combine(duel_dqn_params_, duel_dqn_static_params)
    # (n_agent, 9)
    q_values = jax.vmap(duel_dqn_model_, )(obs)
    q_values_4_selecting = q_values * legal_actions_ + (1.0 - legal_actions_) * -1e6
    joint_actions = jax.lax.cond(
        jnp.less(random_act_prob, eps_rate_),
        lambda _: random_joint_actions,
        lambda _: jnp.argmax(q_values_4_selecting, axis=1),
        None,
    )
    return joint_actions, q_values


def get_q_values(
        updating_duel_dqn_dynamic_params_,
        obs_,
):
    duel_model = eqx.combine(updating_duel_dqn_dynamic_params_, duel_dqn_static_params)
    return jax.vmap(duel_model)(obs_)


# def l2_norm(dynamic_params):
#     model = eqx.combine(dynamic_params, duel_dqn_static_params)
#     total = 0
#     for leaf in jax.tree_util.tree_leaves(model):
#         if eqx.is_inexact_array(leaf):
#             total = total + jnp.sum(leaf ** 2)
#     return total


def collect_trajectory(rollout_inputs, _):
    (
        duel_dqn_dynamic_params,
        greedy_actions_key,
        env_reset_keys,
        env_states,
        observations,
        eps_rate,
    ) = rollout_inputs

    new_greedy_actions_key, *greedy_actions_keys = jax.random.split(greedy_actions_key, 1 + NUM_ENVS)
    greedy_actions_keys = jnp.array(greedy_actions_keys)

    joint_actions, _ = jax.vmap(
        greedy_eps_selection,
        in_axes=(None, 0, 0, 0, None)
    )(
        duel_dqn_dynamic_params,
        observations,
        env_states["action_mask"],
        greedy_actions_keys,
        eps_rate,
    )

    (
        new_env_reset_keys,
        new_env_states,
        rewards,
        terminated,
    ) = jax.vmap(env.step)(
        joint_actions,
        env_states,
        env_reset_keys,
    )
    propagation_maps_from_new_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.full_propagation_maps_train,
        new_env_states["env_map_index"]
    )
    building_maps_from_new_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.environment_building_maps_train,
        new_env_states["env_map_index"]
    )
    locnet_prediction_maps = jax.vmap(locnet_model)(
        jnp.stack(
            [
                propagation_maps_from_new_states * jnp.where(new_env_states["visited_positions"] != 0.0, 1.0, 0.0),
                (0.0 - building_maps_from_new_states) + jnp.where(new_env_states["visited_positions"] != 0.0, 1.0, 0.0),
            ],
            axis=1
        ),
    ).squeeze()

    new_observations = jax.vmap(env.get_obs)(
        new_env_states["robots_positions"],
        propagation_maps_from_new_states,
        building_maps_from_new_states,
        new_env_states["visited_positions"],
        locnet_prediction_maps
    )

    return (
        duel_dqn_dynamic_params,
        new_greedy_actions_key,
        new_env_reset_keys,
        new_env_states,
        new_observations,
        eps_rate,
    ), (
        observations,
        env_states["action_mask"],
        joint_actions,
        rewards,
        terminated,
    )


def loss_fn(
        dual_dqn_dynamic_params,
        current_obs,
        actions_mask,
        picked_actions,
        rewards,
        terminated,
):
    reshaped_obs, reshape_action_mask = (
        jnp.reshape(current_obs, (-1, *current_obs.shape[2:])),
        jnp.reshape(actions_mask, (-1, *actions_mask.shape[2:])),
    )

    # n_steps * batch_size, n_agent, 9
    pred_q_values_from_reshaped = jax.vmap(
        get_q_values,
        in_axes=(None, 0)
    )(
        dual_dqn_dynamic_params,
        reshaped_obs,
    )
    # weight_l2 = l2_norm(dual_dqn_dynamic_params)
    pred_q_values_from_reshaped = pred_q_values_from_reshaped * reshape_action_mask + (1.0 - reshape_action_mask) * -1e6
    pred_q_values = jnp.reshape(pred_q_values_from_reshaped, (BATCH_SIZE, MINI_TRAJ_NUM, NUM_AGENTS, 9))

    chosen_actions_q_vals = jnp.take_along_axis(
        pred_q_values,
        jnp.expand_dims(picked_actions, axis=-1),
        axis=-1,
    ).squeeze()

    chosen_actions_q_vals = jnp.sum(chosen_actions_q_vals, axis=-1).at[:, :-1].get()
    # q_values, rewards, terminated = inputs

    targets_ = get_target(
        (
            pred_q_values,
            rewards,
            terminated,
        )
    )
    loss = 0.5 * jnp.square(chosen_actions_q_vals - jax.lax.stop_gradient(targets_)).mean()
    return loss


def preprocessed_batch(x_, key_):
    x_ = jax.random.permutation(key_, x_)
    x_ = jnp.reshape(
        x_, (NUM_BATCHES, BATCH_SIZE, *x_.shape[1:])
    )
    return x_


@jax.jit
def learn_phase(carry_state, batch):
    updating_dual_dqn_dynamic_params_, updating_opt_state_duel_dqn_ = carry_state
    obs, action_mask, picked_actions, rewards, terminated = batch

    loss_value, grads = jax.value_and_grad(loss_fn)(
        updating_dual_dqn_dynamic_params_,
        obs,
        action_mask,
        picked_actions,
        rewards,
        terminated,
    )
    updated_duel_dqn_model_ = eqx.combine(
        updating_dual_dqn_dynamic_params_,
        duel_dqn_static_params
    )
    updates, updated_opt_state_duel_dqn_ = optim_duel_dqn.update(
        grads, updating_opt_state_duel_dqn_,
        updated_duel_dqn_model_
    )
    updated_duel_dqn_model_ = eqx.apply_updates(
        updated_duel_dqn_model_,
        updates
    )
    updated_dual_dqn_dynamic_params_, dummy__ = eqx.partition(
        updated_duel_dqn_model_,
        eqx.is_array
    )

    return (
        updated_dual_dqn_dynamic_params_,
        updated_opt_state_duel_dqn_
    ), None


def validation_simulation(duel_dqn_dynamic_params, reset_key):
    val_duel_dqn_model = eqx.combine(duel_dqn_dynamic_params, duel_dqn_static_params)
    _, *val_reset_env_keys = jax.random.split(reset_key, 1 + NUM_VAL_SIMULATIONS)
    val_reset_env_keys = jnp.array(val_reset_env_keys)
    _, env_states = jax.vmap(env.validation_env_reset)(val_reset_env_keys)

    propagation_maps_from_env_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.full_propagation_maps_val,
        env_states["env_map_index"]
    )
    building_maps_from_env_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.environment_building_maps_val,
        env_states["env_map_index"]
    )

    locnet_predictions_from_env_states = jax.vmap(locnet_model)(
        jnp.stack(
            [
                propagation_maps_from_env_states * jnp.where(env_states["visited_positions"] != 0.0, 1.0, 0.0),
                (0.0 - building_maps_from_env_states) + jnp.where(env_states["visited_positions"] != 0.0, 1.0, 0.0),
            ],
            axis=1,
        ),
    ).squeeze()

    env_observations = jax.vmap(env.get_obs)(
        env_states["robots_positions"],
        propagation_maps_from_env_states,
        building_maps_from_env_states,
        env_states["visited_positions"],
        locnet_predictions_from_env_states
    )

    def stop_func_(x):
        return jnp.logical_not(jnp.all(x[-1]))

    def rollout(val_rollout_inputs):
        (
            states,
            observations,
            is_successes,
            time_takes,
            action_takes,
            terminations,
        ) = val_rollout_inputs
        # val_obs: n_envs, n_agents, obs ...
        # n_envs * n_agents, obs
        observations = jnp.reshape(
            observations,
            (-1, *observations.shape[2:])
        )
        action_mask = jnp.reshape(
            states["action_mask"],
            (-1, *states["action_mask"].shape[2:])
        )

        joint_actions = jax.vmap(val_duel_dqn_model)(observations)
        joint_actions = jnp.where(action_mask == 0, -1e6, joint_actions)
        joint_actions = jnp.argmax(joint_actions, axis=-1)
        joint_actions = jnp.reshape(joint_actions, (NUM_VAL_SIMULATIONS, NUM_AGENTS))

        new_states, new_terminations, new_successes, new_times_take, new_actions_take = jax.vmap(env.val_step)(
            joint_actions,
            states,
            terminations,
            is_successes,
            time_takes,
            action_takes,
        )

        propagation_maps_from_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
            env.full_propagation_maps_val,
            new_states["env_map_index"]
        )
        building_maps_from_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
            env.environment_building_maps_val,
            new_states["env_map_index"]
        )

        locnet_predictions_from_states = jax.vmap(locnet_model)(
            jnp.stack(
                [
                    propagation_maps_from_states * jnp.where(new_states["visited_positions"] != 0.0, 1.0, 0.0),
                    (0.0 - building_maps_from_states) + jnp.where(new_states["visited_positions"] != 0.0, 1.0, 0.0),
                ],
                axis=1,
            ),
        ).squeeze()

        new_observations = jax.vmap(env.get_obs)(
            new_states["robots_positions"],
            propagation_maps_from_states,
            building_maps_from_states,
            new_states["visited_positions"],
            locnet_predictions_from_states
        )

        return (
            new_states,
            new_observations,
            new_successes,
            new_times_take,
            new_actions_take,
            new_terminations,
        )

    _, _, total_success, total_times_take, total_actions_take, _ = jax.lax.while_loop(
        stop_func_,
        rollout,
        (
            env_states,
            env_observations,
            jnp.zeros(shape=(NUM_VAL_SIMULATIONS,), dtype=jnp.bool_),
            jnp.zeros(shape=(NUM_VAL_SIMULATIONS,), dtype=jnp.int32),
            jnp.zeros(shape=(NUM_VAL_SIMULATIONS,), dtype=jnp.int32),
            jnp.zeros(shape=(NUM_VAL_SIMULATIONS,), dtype=jnp.bool_),
        )
    )
    return total_success, total_times_take, total_actions_take


@jax.jit
def update_duel_dqn(update_inputs, _):
    (
        dual_dqn_dynamic_params,
        dual_dqn_opt_state,
        observations,
        action_masks,
        actions,
        rewards,
        terminations,
        shuffling_key,
    ) = update_inputs
    shuffling_key, batch_shuffling_key_ = jax.random.split(shuffling_key, 1 + 1)

    batched_observations = preprocessed_batch(jnp.transpose(observations, (1, 0, 2, 3, 4, 5)), batch_shuffling_key_)
    batched_action_masks = preprocessed_batch(jnp.transpose(action_masks, (1, 0, 2, 3)), batch_shuffling_key_)
    batched_actions = preprocessed_batch(jnp.transpose(actions, (1, 0, 2)), batch_shuffling_key_)
    batched_rewards = preprocessed_batch(jnp.transpose(rewards, (1, 0)), batch_shuffling_key_)
    batched_terminations = preprocessed_batch(jnp.transpose(terminations, (1, 0)), batch_shuffling_key_)

    (
        updated_dual_dqn_dynamic_params,
        updated_dual_dqn_opt_state
    ), _ = jax.lax.scan(
        learn_phase,
        (
            dual_dqn_dynamic_params,
            dual_dqn_opt_state
        ),
        (
            batched_observations,
            batched_action_masks,
            batched_actions,
            batched_rewards,
            batched_terminations,
        )
    )

    return (
        updated_dual_dqn_dynamic_params,
        updated_dual_dqn_opt_state,
        observations,
        action_masks,
        actions,
        rewards,
        terminations,
        shuffling_key,
    ), _


def train(inputs, _):
    (
        misc_key,
        env_reset_keys,
        env_states,
        env_observations,
        duel_dqn_dynamic_params,
        opt_state_duel_dqn_,
        eps_rate,
        validation_count,
        validation_success_rate_checkpoints_value,
    ) = inputs

    # STEP
    new_misc_key, validation_sim_key, action_key, batch_shuffling_key = jax.random.split(misc_key, 1 + 1 + 1 + 1)
    (
        _, _,
        new_env_reset_keys,
        new_env_states,
        new_env_observations,
        _,
    ), (
        observations_buffers,
        actions_masks_buffers,
        joint_actions_buffers,
        rewards_buffers,
        termination_buffers,
    ) = jax.lax.scan(
        collect_trajectory,
        (
            duel_dqn_dynamic_params,
            action_key,
            env_reset_keys,
            env_states,
            env_observations,
            eps_rate,
        ),
        None,
        MINI_TRAJ_NUM
    )
    # UPDATE
    (
        new_duel_dqn_dynamic_params,
        new_opt_state_duel_dqn,
        _, _, _, _, _, _,
    ), _ = jax.lax.scan(
        update_duel_dqn,
        (
            duel_dqn_dynamic_params,
            opt_state_duel_dqn_,
            observations_buffers,
            actions_masks_buffers,
            joint_actions_buffers,
            rewards_buffers,
            termination_buffers,
            batch_shuffling_key,
        ),
        None,
        NUM_EPOCHS
    )
    # VALIDATE
    total_success_cases, total_times_take, total_actions_take = validation_simulation(new_duel_dqn_dynamic_params, validation_sim_key)
    total_success_cases, total_times_take, total_actions_take = jnp.mean(total_success_cases, keepdims=True), jnp.mean(total_times_take, keepdims=True), jnp.mean(total_actions_take, keepdims=True)

    def make_check_point():
        def callback(callback_inputs):
            saved_actor_params, sum_val_rewards, times_takes, actions_takes, val_count_ = callback_inputs
            saved_actor = eqx.combine(saved_actor_params, duel_dqn_static_params)
            writer.add_scalar("Success_rate", sum_val_rewards.item(), val_count_.item())
            writer.add_scalar("Localization_time", times_takes.item(), val_count_.item())
            writer.add_scalar("Actions", actions_takes.item(), val_count_.item())
            writer.flush()
            eqx.tree_serialise_leaves(
                f"/home/Users/PQLocNet_No_Dueling/SOTA/"
                f"PQLOCNET_DENSE_REWARDS_4_ROBOTS_CLIP_BY_5_LR_5e4_50M_CLONE_9_NO_DUELING_SPARSE_{val_count_.item()}_"
                f"SUC_{sum_val_rewards.item()}_LOC_{times_takes.item()}_ACTION_{actions_takes.item()}.eqx",
                saved_actor
            )
            pBar.update(val_count_.item(), values=[("Success cumulate", sum_val_rewards.item())])

        jax.experimental.io_callback(
            callback,
            None,
            (new_duel_dqn_dynamic_params, total_success_cases, total_times_take, total_actions_take, validation_count)
        )
        return total_success_cases

    def logging():
        def log_callback(log_callback_inputs):
            sum_val_rewards, times_takes, actions_takes, val_count_ = log_callback_inputs
            writer.add_scalar("Success_rate", sum_val_rewards.item(), val_count_.item())
            writer.add_scalar("Localization_time", times_takes.item(), val_count_.item())
            writer.add_scalar("Actions", actions_takes.item(), val_count_.item())
            writer.flush()
            pBar.update(val_count_.item(), values=[("Success cumulate", sum_val_rewards.item())])

        jax.experimental.io_callback(
            log_callback,
            None,
            (total_success_cases, total_times_take, total_actions_take, validation_count)
        )
        return validation_success_rate_checkpoints_value

    validation_success_rate_checkpoints_value = jax.lax.cond(
        jnp.less_equal(validation_success_rate_checkpoints_value, total_success_cases).sum(),
        lambda _: make_check_point(),
        lambda _: logging(),
        None
    )

    validation_count += 1
    eps_rate = jax.lax.cond(
        jnp.less_equal(eps_rate, 0.01).sum(),
        lambda _: 0.01,
        lambda _: eps_rate - 0.0004,
        None
    )

    return (
        new_misc_key,
        new_env_reset_keys,
        new_env_states,
        new_env_observations,
        new_duel_dqn_dynamic_params,
        new_opt_state_duel_dqn,
        eps_rate,
        validation_count,
        validation_success_rate_checkpoints_value,
    ), _


@eqx.filter_jit
def make_train(env_keys, misc_key):
    reset_env_keys, env_states = jax.vmap(env.reset)(env_keys)

    propagation_maps_from_env_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.full_propagation_maps_train,
        env_states["env_map_index"]
    )

    building_maps_from_env_states = jax.vmap(extracted_maps, in_axes=(None, 0))(
        env.environment_building_maps_train,
        env_states["env_map_index"]
    )

    locnet_predictions = jax.vmap(locnet_model)(
        jnp.stack(
            [
                propagation_maps_from_env_states * jnp.where(env_states["visited_positions"] != 0.0, 1.0, 0.0),
                (0.0 - building_maps_from_env_states) + jnp.where(env_states["visited_positions"] != 0.0, 1.0, 0.0),
            ],
            axis=1
        ),
    ).squeeze()

    env_observations = jax.vmap(env.get_obs)(
        env_states["robots_positions"],
        propagation_maps_from_env_states,
        building_maps_from_env_states,
        env_states["visited_positions"],
        locnet_predictions,
    )

    _, _ = jax.lax.scan(
        train,
        (
            misc_key,
            reset_env_keys,
            env_states,
            env_observations,
            duel_dqn_params,
            opt_state_duel_dqn,
            EPS_RATE,
            jnp.zeros(shape=()),
            jnp.array([-1 * jnp.inf]),
        ),
        None,
        int(MAX_TRAINING_STEPS / (MINI_TRAJ_NUM * NUM_ENVS))
    )


if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    locnet_model = LocNet(
        2,
        4,
        4,
        1,
        jax.random.PRNGKey(0),
        jax.random.PRNGKey(1),
        27,
        0.3
    )
    locnet_model = eqx.tree_deserialise_leaves(
        "William_LocNet.eqx",
        locnet_model
    )

    locnet_model = eqx.nn.inference_mode(locnet_model)

    pBar = keras.utils.Progbar(target=int(MAX_TRAINING_STEPS / (MINI_TRAJ_NUM * NUM_ENVS)), verbose=1)
    dpms_list_train, buildings_list_train, floodfill_list_train = get_all_maps(
        path_to_DPM_imgs='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/DPM',
        path_to_building_imgs='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/buildings',
        path_to_floodfill_data='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/Train',
    )

    dpms_list_val, buildings_list_val, floodfill_list_val = get_all_maps(
        path_to_DPM_imgs='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/Val_DPM',
        path_to_building_imgs='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/Val_buildings',
        path_to_floodfill_data='/home/Users/DQN_LOCNET/Five_antennas_per_env_100_envs/Val',
    )

    env = Environment(
        max_time_step=MAX_TIME,
        n_agents=NUM_AGENTS,
        m_per_step=M_PER_STEPS,
        minimum_distance_deployment_from_transmitter=MINIMUM_DISTANCE_DEPLOYMENT,
        full_propagation_maps_train=dpms_list_train,
        building_maps_train=buildings_list_train,
        shortest_distance_map_train=floodfill_list_train,
        full_propagation_maps_val=dpms_list_val,
        building_maps_val=buildings_list_val,
        shortest_distance_map_val=floodfill_list_val,
        distance_within_to_terminate=DISTANCE_WITHIN_2_TERMINATE,
    )
    key = jax.random.PRNGKey(KEY_SEED)
    misc_key_g, dueling_dqn_key, *env_keys_g = jax.random.split(key, 1 + 1 + NUM_ENVS)
    env_keys_g = jnp.array(env_keys_g)

    duel_dqn_model = DuelingDQN(
        observation_space_dims=6,
        action_space_dims=9,
        key=dueling_dqn_key,
    )

    # optim_duel_dqn = optax.radam(learning_rate=LR_DUEL_DQN)
    optim_duel_dqn = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIPPING),
        optax.radam(learning_rate=LR_DUEL_DQN)
    )

    opt_state_duel_dqn = optim_duel_dqn.init(eqx.filter(duel_dqn_model, eqx.is_array))
    duel_dqn_params, duel_dqn_static_params = eqx.partition(duel_dqn_model, eqx.is_array)

    writer = SummaryWriter(comment="CLIPPING_4_AGENTS_CLIP_BY_5_LR_5e4_50M_DENSE_CLONE_9_NO_DUELING_SPARSE")
    make_train(env_keys_g, misc_key_g)
    writer.close()
