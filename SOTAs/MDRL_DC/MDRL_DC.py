# All jax libraries
import cv2
import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
import jax.numpy as jnp
import keras
import equinox as eqx
import optax
from functools import partial
from Environment_4 import Environment
from Actor_Critic_Models_2 import Actor, Critic
from utils import get_all_maps
import distrax
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from CONSTANTS import *


@jax.jit
def extracted_maps(
        array,
        index
):
    return array.at[index].get()


# def fixing_logits(student_policy_mask_out, expert_action_logits):
#     return jax.lax.cond(
#         jnp.all(jnp.where(student_policy_mask_out == -1e6, True, False)),
#         lambda _: expert_action_logits,
#         lambda _: student_policy_mask_out,
#         None
#     )


# WHATEVER IN HERE (N_AGENTS, OBSERVATIONS_DIM)
@eqx.filter_jit
def get_action(
        actor_model_,
        expert,
        obs,
        env_building_embedding_,
        legal_actions_,
        sample_key,
        action_prob_threshold,
        expert_rate
):
    def get_joint_actions_from_expert(student_joint_actions):
        student_joint_action_masks = jnp.where(student_joint_actions > action_prob_threshold, True, False)
        observations_4_expert = obs.at[:, :-1].get()
        expert_actions_logits_org = jax.vmap(expert, in_axes=(0, None))(
            observations_4_expert,
            env_building_embedding_
        )
        # JUST 2 BE SAFE
        expert_actions_logits_action_mask = jnp.where(legal_actions_, expert_actions_logits_org, -1e8)

        expert_actions_logits_student_policy_mask = jnp.where(
            student_joint_action_masks,
            expert_actions_logits_action_mask,
            -1e8
        )
        expert_actions_logits = expert_actions_logits_student_policy_mask
        # expert_actions_logits = jax.vmap(fixing_logits)(
        #     expert_actions_logits_student_policy_mask,
        #     expert_actions_logits_action_mask
        # )
        distribution_4_expert = distrax.Categorical(logits=expert_actions_logits)
        return distribution_4_expert.mode().astype(jnp.int32)

    sample_key, is_using_expert_key = jax.random.split(sample_key, 1 + 1)
    is_using_expert = jax.random.uniform(is_using_expert_key)

    action_logits = jax.vmap(
        actor_model_,
        in_axes=(0, None)
    )(obs, env_building_embedding_)

    action_logits = jnp.where(legal_actions_, action_logits, -1e8)
    dist = distrax.Categorical(logits=action_logits)

    joint_actions = jax.lax.cond(
        jnp.greater_equal(expert_rate, is_using_expert),
        lambda _: get_joint_actions_from_expert(dist.probs),
        lambda _: dist.sample(seed=sample_key, sample_shape=(1,)).squeeze().astype(jnp.int32),
        None
    )

    return joint_actions, dist.log_prob(joint_actions)


@eqx.filter_jit
def calc_gae(
        critic_values,
        rewards,
        terminate,
        gamma,
        gamma_lambda
):
    def calc_advantage(gae_and_next_value, transition_per_step):
        gae_, next_value = gae_and_next_value
        done, value, reward = transition_per_step
        delta = reward + gamma * next_value * (1 - done) - value
        gae_ = delta + gamma * gamma_lambda * (1 - done) * gae_
        return (gae_, value), gae_

    _, advantages_ = jax.lax.scan(
        calc_advantage,
        (jnp.zeros_like(critic_values.at[-1].get()), critic_values.at[-1].get()),
        (terminate, critic_values, rewards),
        reverse=True,
        unroll=16,
    )
    returns = critic_values + advantages_
    advantages_ = (advantages_ - jnp.mean(advantages_, axis=0, keepdims=True)) / (
            jnp.std(advantages_, axis=0, keepdims=True) + jnp.finfo(advantages_.dtype).eps)
    return advantages_, returns


@eqx.filter_jit
def make_step_actor(
        actor_m,
        critic_m,
        opt_act,
        obs,
        picked_actions,
        picked_log_probs,
        rewards,
        terminated,
        building_embedding_,
        gamma, gamma_lambda, entropy_coef, clip_eps
):
    loss, grads = eqx.filter_value_and_grad(actor_loss)(
        actor_m,
        critic_m,
        obs,
        picked_actions,
        picked_log_probs,
        rewards,
        terminated,
        building_embedding_,
        gamma, gamma_lambda, entropy_coef, clip_eps
    )
    updates, opt_state = optim_actor.update(grads, opt_act, actor_m)
    actor_m = eqx.apply_updates(actor_m, updates)
    return loss, actor_m, opt_state


@eqx.filter_jit
def get_preds(
        model,
        obs,
        building_embedding,
):
    return jax.vmap(model, in_axes=(0, None))(obs, building_embedding)


@eqx.filter_jit
def actor_loss(
        actor_m,
        critic_m,
        obs,
        picked_actions,
        picked_log_probs,
        rewards,
        terminated,
        building_embedding_,
        gamma, gamma_lambda, entropy_coef, clip_eps
):
    new_logits = jax.vmap(get_preds, in_axes=(None, 0, 0))(actor_m, obs, building_embedding_)
    dist = distrax.Categorical(logits=new_logits)
    new_critic_values_ = jax.vmap(get_preds, in_axes=(None, 0, 0))(critic_m, obs, building_embedding_).squeeze()
    new_picked_log_probs = dist.log_prob(picked_actions)
    entropy = dist.entropy()

    new_critic_values_ = jnp.reshape(new_critic_values_, (MINI_TRAJ_SIZE, NUM_ENVS, *new_critic_values_.shape[1:]))
    advantages, _ = calc_gae(
        new_critic_values_,
        rewards,
        terminated,
        gamma,
        gamma_lambda
    )

    advantages = jnp.reshape(advantages, (-1, *advantages.shape[2:]))
    ratio = jnp.exp(new_picked_log_probs - picked_log_probs)
    loss_actor1 = ratio * advantages

    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    return loss_actor.mean() - entropy_coef * entropy.mean()


@eqx.filter_jit
def critic_loss(
        critic_m,
        obs,
        rewards,
        terminated,
        building_embedding,
        gamma,
        gamma_lambda,
        vf_coef
):
    new_critic_values = jax.vmap(get_preds, in_axes=(None, 0, 0))(critic_m, obs, building_embedding).squeeze()
    new_critic_values = jnp.reshape(new_critic_values, (MINI_TRAJ_SIZE, NUM_ENVS, *new_critic_values.shape[1:]))
    _, targets = calc_gae(
        new_critic_values,
        rewards,
        terminated,
        gamma,
        gamma_lambda
    )
    new_critic_values_ = jnp.reshape(new_critic_values, (-1, *new_critic_values.shape[2:]))
    targets = jnp.reshape(targets, (-1, *targets.shape[2:]))
    targets = jax.lax.stop_gradient(targets)

    # calc critic loss
    values_losses = jnp.square(new_critic_values_ - targets)
    value_loss = values_losses.mean()
    return vf_coef * value_loss


@eqx.filter_jit
def make_step_critic(
        critic_m,
        opt_cri,
        obs,
        rewards,
        terminated,
        building_embedding_,
        gamma, gamma_lambda, vf_coef
):
    loss, grads = eqx.filter_value_and_grad(critic_loss)(
        critic_m,
        obs,
        rewards,
        terminated,
        building_embedding_,
        gamma, gamma_lambda, vf_coef
    )
    updates, opt_state = optim_actor.update(grads, opt_cri, critic_m)
    critic_m = eqx.apply_updates(critic_m, updates)
    return loss, critic_m, opt_state


@eqx.filter_jit
def get_deterministic_actions_policy(
        actor,
        state,
        building_embedding
):
    joint_actions_logits = jax.vmap(actor, in_axes=(0, None))(state["obs"], building_embedding)
    joint_actions_logits = jnp.where(state["action_mask"], joint_actions_logits, -1e8)
    distribution = distrax.Categorical(logits=joint_actions_logits)
    joint_actions = distribution.mode().astype(jnp.int32)
    return joint_actions


@eqx.filter_jit
def validation_simulation(
        actor_params,
        reset_key,
        buildings_embeddings,
        num_val_simulations
):
    actor = eqx.combine(actor_params, actor_static_params)
    _, initial_states = jax.vmap(env.validation_env_reset)(reset_key)

    def stop_func(x):
        return jnp.logical_not(jnp.all(x[2]))

    def rollout(carry):
        states, b_embeddings, terminations, success, time_steps, total_actions = carry
        current_building_embeddings = jax.vmap(extracted_maps, in_axes=(None, 0))(b_embeddings, states["env_map_index"])
        joint_actions = jax.vmap(
            get_deterministic_actions_policy,
            in_axes=(None, 0, 0)
        )(actor, states, current_building_embeddings)

        new_states, new_terminations, new_success, new_time_steps, new_total_actions = jax.vmap(env.val_step)(
            joint_actions,
            states,
            terminations,
            success,
            time_steps,
            total_actions,
        )

        return (
            new_states,
            b_embeddings,
            new_terminations,
            new_success,
            new_time_steps,
            new_total_actions
        )

    _, _, _, total_success_simulations, total_time_steps, total_actions_taken = jax.lax.while_loop(
        stop_func,
        rollout,
        (
            initial_states,
            buildings_embeddings,
            jnp.zeros((num_val_simulations,), dtype=jnp.bool_),
            jnp.zeros((num_val_simulations,), dtype=jnp.int32),
            jnp.zeros((num_val_simulations,), dtype=jnp.int32),
            jnp.zeros((num_val_simulations,), dtype=jnp.int32),
        )
    )
    return total_success_simulations, total_time_steps, total_actions_taken


@eqx.filter_jit
def mini_traj_rollout(rollout_inputs, _, num_envs, action_prob_threshold, expert_rate):
    actor_params, expert_params, sampled_key, continuous_reset_env_keys_from_rollout, continuous_env_states_from_rollout = rollout_inputs
    actor = eqx.combine(actor_params, actor_static_params)
    expert = eqx.combine(expert_params, expert_static_params)
    new_sampled_key, *sampled_keys = jax.random.split(sampled_key, 1 + num_envs)
    sampled_keys = jnp.array(sampled_keys)
    b_embedding_train = jax.vmap(extracted_maps, in_axes=(None, 0))(
        building_embeddings_train,
        continuous_env_states_from_rollout["env_map_index"]
    )
    # CHOPPED OF ENVS DIMENSION
    current_joint_actions, current_log_action_logits = jax.vmap(
        get_action,
        in_axes=(None, None, 0, 0, 0, 0, None, None)
    )(
        actor,
        expert,
        continuous_env_states_from_rollout["obs"],
        b_embedding_train,
        continuous_env_states_from_rollout["action_mask"],
        sampled_keys,
        action_prob_threshold,
        expert_rate,
    )

    (
        continuous_next_reset_env_keys_from_rollout,
        continuous_next_env_states_from_rollout,
        rewards,
        terminated,
    ) = jax.vmap(env.step)(
        current_joint_actions,
        continuous_env_states_from_rollout,
        continuous_reset_env_keys_from_rollout,
    )

    return (
        actor_params,
        expert_params,
        new_sampled_key,
        continuous_next_reset_env_keys_from_rollout,
        continuous_next_env_states_from_rollout,
    ), (
        continuous_env_states_from_rollout["obs"],
        current_joint_actions,
        rewards,
        terminated,
        current_log_action_logits,
        b_embedding_train,
    )


def update_actor_critic(update_inputs, _, gamma, gamma_lambda, entropy_coef, clip_eps, vf_coef):
    (
        actor_params,
        critic_params,
        opt_actor,
        opt_critic,
        current_obs,
        picked_actions,
        picked_log_probs,
        rewards,
        terminates_,
        building_embedding,
    ) = update_inputs

    obs_4_loss = jnp.reshape(current_obs, (-1, *current_obs.shape[2:]))
    picked_actions_4_loss = jnp.reshape(picked_actions, (-1, *picked_actions.shape[2:]))
    picked_log_probs_4_loss = jnp.reshape(picked_log_probs, (-1, *picked_log_probs.shape[2:]))
    building_embedding_4_loss = jnp.reshape(building_embedding, (-1, *building_embedding.shape[2:]))

    updating_actor = eqx.combine(actor_params, actor_static_params)
    updating_critic = eqx.combine(critic_params, critic_static_params)

    act_loss, new_actor, new_opt_actor = make_step_actor(
        updating_actor,
        updating_critic,
        opt_actor,
        obs_4_loss,
        picked_actions_4_loss,
        picked_log_probs_4_loss,
        rewards,
        terminates_,
        building_embedding_4_loss,
        gamma, gamma_lambda, entropy_coef, clip_eps
    )

    cri_loss, new_critic, new_opt_critic = make_step_critic(
        updating_critic,
        opt_critic,
        obs_4_loss,
        rewards,
        terminates_,
        building_embedding_4_loss,
        gamma, gamma_lambda, vf_coef
    )

    new_actor_params, _ = eqx.partition(new_actor, eqx.is_array)
    new_critic_params, _ = eqx.partition(new_critic, eqx.is_array)

    return (
        new_actor_params,
        new_critic_params,
        new_opt_actor,
        new_opt_critic,
        current_obs,
        picked_actions,
        picked_log_probs,
        rewards,
        terminates_,
        building_embedding
    ), None


@eqx.filter_jit
def train_model(
        inputs,
        _,
        gamma,
        gamma_lambda,
        entropy_coef,
        clip_eps,
        vf_coef,
        num_envs,
        num_simulations_val,
        action_prob_threshold,
        expert_rate
):
    (
        sample_action_key,
        continuous_reset_env_keys,
        validation_sim_key,
        continuous_env_states,
        actor_params,
        critic_params,
        expert_params,
        opt_state_actor_,
        opt_state_critic_,
        val_count,
        prev_cum_reward,
    ) = inputs
    validation_sim_key, *validation_sim_keys = jax.random.split(validation_sim_key, 1 + num_simulations_val)
    validation_sim_keys = jnp.array(validation_sim_keys)
    (
        _, _,
        next_sample_action_key_,
        next_continuous_reset_env_keys_,
        next_continuous_env_states_,
    ), (
        current_obs_experience_,
        current_joint_actions_,
        rewards_,
        terminated_,
        log_probs_,
        current_building_embeddings_,
    ) = jax.lax.scan(
        partial(mini_traj_rollout, num_envs=num_envs, action_prob_threshold=action_prob_threshold,
                expert_rate=expert_rate),
        (
            actor_params,
            expert_params,
            sample_action_key,
            continuous_reset_env_keys,
            continuous_env_states,
        ),
        None,
        MINI_TRAJ_SIZE
    )

    (
        new_actor_dynamic_params,
        new_critic_dynamic_params,
        new_opt_state_actor,
        new_opt_state_critic,
        _, _, _, _, _, _
    ), _ = jax.lax.scan(
        partial(
            update_actor_critic,
            gamma=gamma,
            gamma_lambda=gamma_lambda,
            entropy_coef=entropy_coef,
            clip_eps=clip_eps,
            vf_coef=vf_coef
        ),
        (
            actor_params,
            critic_params,
            opt_state_actor_,
            opt_state_critic_,
            current_obs_experience_,
            current_joint_actions_,
            log_probs_,
            rewards_,
            terminated_,
            current_building_embeddings_,
        ),
        None,
        NUM_EPOCHS
    )

    total_val_rewards, total_times, total_actions = validation_simulation(
        new_actor_dynamic_params,
        validation_sim_keys,
        building_embeddings_val,
        num_simulations_val
    )
    total_val_rewards = jax.block_until_ready(jnp.mean(total_val_rewards, keepdims=True))
    total_times = jax.block_until_ready(jnp.mean(total_times, keepdims=True))
    total_actions = jax.block_until_ready(jnp.mean(total_actions, keepdims=True))

    def make_check_point(actor_dynamic_params__, total_val_rewards_, total_times_, total_actions_, val_count_):
        def callback(callback_inputs):
            saved_actor_params, sum_val_rewards, times_, actions_, val_count__ = callback_inputs
            saved_actor = eqx.combine(saved_actor_params, actor_static_params)
            writer.add_scalar("Success_rate", sum_val_rewards.item(), val_count__.item())
            writer.add_scalar("Localization_time", times_.item(), val_count__.item())
            writer.add_scalar("Actions", actions_.item(), val_count__.item())
            writer.flush()
            pBar.update(val_count__.item(), values=[("success rate", sum_val_rewards.item())])
            eqx.tree_serialise_leaves(
                f"/home/Users/DC_SOTA_FINAL/SOTA/"
                f"DC_PPO_FINAL_1e6_CLONE_12_{val_count__.item()}_PROPOSAL_"
                f"SUC_{sum_val_rewards.item()}_"
                f"TIMES_{times_.item()}_"
                f"ACTIONS_{actions_.item()}.eqx",
                saved_actor
            )
            return None

        jax.experimental.io_callback(
            callback,
            None,
            (actor_dynamic_params__, total_val_rewards_, total_times_, total_actions_, val_count_)
        )
        return total_val_rewards

    def logging(total_val_rewards_, total_times_, total_actions_, val_count_):
        def log_callback(log_callback_inputs):
            sum_val_rewards, times_, actions_, val_count__ = log_callback_inputs
            writer.add_scalar("Success_rate", sum_val_rewards.item(), val_count__.item())
            writer.add_scalar("Localization_time", times_.item(), val_count__.item())
            writer.add_scalar("Actions", actions_.item(), val_count__.item())
            writer.flush()
            pBar.update(val_count__.item(), values=[("success rate", sum_val_rewards.item())])
            return None

        jax.experimental.io_callback(log_callback, None, (total_val_rewards_, total_times_, total_actions_, val_count_))
        return prev_cum_reward

    prev_cum_reward = jax.lax.cond(
        jnp.less_equal(prev_cum_reward, total_val_rewards).sum(),
        lambda _: make_check_point(new_actor_dynamic_params, total_val_rewards, total_times, total_actions, val_count),
        lambda _: logging(total_val_rewards, total_times, total_actions, val_count),
        None
    )

    val_count += 1
    return (
        next_sample_action_key_,
        next_continuous_reset_env_keys_,
        validation_sim_key,
        next_continuous_env_states_,
        new_actor_dynamic_params,
        new_critic_dynamic_params,
        expert_params,
        new_opt_state_actor,
        new_opt_state_critic,
        val_count,
        prev_cum_reward,
    ), _


@eqx.filter_jit
def create_agent_env_n_train(
        gamma,
        gamma_lambda,
        entropy_coef,
        clip_eps,
        vf_coef,
        num_envs,
        num_simulations_val,
        max_training_steps,
        mini_traj_size,
        action_prob_threshold,
        expert_rate,
        key_action,
        env_keys,
        val_sim_key,
):
    reset_env_keys, env_states = jax.vmap(env.reset)(env_keys)

    _, _ = jax.lax.scan(
        partial(
            train_model,
            gamma=gamma,
            gamma_lambda=gamma_lambda,
            entropy_coef=entropy_coef,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            num_envs=num_envs,
            num_simulations_val=num_simulations_val,
            action_prob_threshold=action_prob_threshold,
            expert_rate=expert_rate
        ),
        (
            key_action,
            reset_env_keys,
            val_sim_key,
            env_states,
            actor_dynamic_params,
            critic_dynamic_params,
            expert_dy_params,
            opt_state_actor,
            opt_state_critic,
            jnp.zeros(shape=()),
            jnp.array([-1 * jnp.inf]),
        ),
        None,
        # jnp.arange(int(max_training_steps / (mini_traj_size * num_envs))),
        int(max_training_steps / (mini_traj_size * num_envs))
    )


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    pBar = keras.utils.Progbar(target=int(MAX_TRAINING_STEPS / (MINI_TRAJ_SIZE * NUM_ENVS)), verbose=1)
    expert_model = Actor(
        observation_space_dims=7,
        action_space_dims=9,
        key=jax.random.PRNGKey(938)
    )
    expert_model = eqx.tree_deserialise_leaves("Expert.eqx", expert_model)
    expert_model = eqx.tree_inference(expert_model)
    expert_dy_params, expert_static_params = eqx.partition(expert_model, eqx.is_array)

    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
    dpms_list_train, buildings_list_train, floodfill_list_train, building_embeddings_train = get_all_maps(
        path_to_DPM_imgs='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/DPM',
        path_to_building_imgs='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/buildings',
        path_to_floodfill_data='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Train',
        path_to_buildings_embeddings='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Train_buildings_embeddings',
    )

    dpms_list_val, buildings_list_val, floodfill_list_val, building_embeddings_val = get_all_maps(
        path_to_DPM_imgs='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Val_DPM',
        path_to_building_imgs='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Val_buildings',
        path_to_floodfill_data='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Val',
        path_to_buildings_embeddings='/home/User/DQN_LOCNET/Five_antennas_per_env_100_envs/Val_buildings_embeddings',
    )

    key = jax.random.PRNGKey(KEY_SEED)
    key_action_, val_sim_key_, critic_key, actor_key, *env_keys_ = jax.random.split(
        key,
        1 + 1 + 1 + 1 + NUM_ENVS
    )
    env_keys_ = jnp.array(env_keys_)
    env = Environment(
        max_time_step=MAX_TIME,
        n_agents=NUM_AGENTS,
        m_per_step=M_PER_STEPS,
        minimum_distance_deployment_from_transmitter=MINIMUM_DEPLOYMENT_FROM_TRANSMITTER,
        full_propagation_maps_train=dpms_list_train,
        building_maps_train=buildings_list_train,
        shortest_distance_map_train=floodfill_list_train,
        full_propagation_maps_val=dpms_list_val,
        building_maps_val=buildings_list_val,
        shortest_distance_map_val=floodfill_list_val,
        distance_within_to_terminate=DISTANCE_WITHIN_TO_TERMINATE,
        building_embedding_train=building_embeddings_train,
        building_embedding_val=building_embeddings_val,
    )
    actor_model = Actor(
        observation_space_dims=8,
        action_space_dims=9,
        key=actor_key,
    )
    critic_model = Critic(
        observation_space_dims=8,
        key=critic_key,
    )

    optim_actor = optax.adam(learning_rate=LR_ACTOR)
    optim_critic = optax.adam(learning_rate=LR_CRITIC)
    opt_state_actor = optim_actor.init(eqx.filter(actor_model, eqx.is_array))
    opt_state_critic = optim_critic.init(eqx.filter(critic_model, eqx.is_array))
    actor_dynamic_params, actor_static_params = eqx.partition(actor_model, eqx.is_array)
    critic_dynamic_params, critic_static_params = eqx.partition(critic_model, eqx.is_array)

    writer = SummaryWriter(comment="DC_CLONE_1e6_CLONE_12")
    jax.block_until_ready(create_agent_env_n_train)(
        GAMMA,
        GAMMA_LAMBDA,
        ENTROPY_COEF,
        CLIP_EPS,
        VF_COEF,
        NUM_ENVS,
        NUM_VAL_SIMULATIONS,
        MAX_TRAINING_STEPS,
        MINI_TRAJ_SIZE,
        ACTION_PROB_THRESHOLD,
        EXPERT_RATE,
        key_action_,
        env_keys_,
        val_sim_key_,
    )
    writer.close()
