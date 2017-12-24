import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std #, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def train(env, num_timesteps, nb_trials, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, test_interval, batch_size, memory, output, load_file,
    save=False, tau=0.01, evaluation=False, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    observation_range=[env.observation_space.low, env.observation_space.high]
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, observation_range=observation_range)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    trial_return_history = deque(maxlen=100)
    eval_trial_return_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #tf.summary.FileWriter(dir_path, sess.graph)

        trial = 0
        ts = 0

        if load_file != '':
          saver.restore(sess, load_file)

        start_time = time.time()

        trial_returns = []
        trial_steps = []
        actions = []
        qs = []
        train_actor_losses = []
        train_critic_losses = []
        train_adaptive_distances = []

        while True:
            test = (test_interval >= 0 and trial%(test_interval+1) == test_interval)

            if not test:
                # Perform rollout.
                env.set_test(test=False)
                obs = env.reset()
                agent.reset()
                done = 0
                trial_return = 0.
                trial_step = 0
                while done == 0:
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    ts += 1
                    if rank == 0 and render:
                        env.render()
                    trial_return += r
                    trial_step += 1

                    # Book-keeping.
                    actions.append(action)
                    qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done == 2) # terminal indicator is 2
                    obs = new_obs

                    # Train.
                    if memory.nb_entries >= batch_size:
                        for t_train in range(nb_train_steps):
                            # Adapt param noise, if necessary.
                            if trial % param_noise_adaption_interval == 0:
                                distance = agent.adapt_param_noise()
                                train_adaptive_distances.append(distance)

                            cl, al = agent.train()
                            train_critic_losses.append(cl)
                            train_actor_losses.append(al)
                            agent.update_target_net()

                # Episode done.
                trial_steps.append(trial_step)
                trial_returns.append(trial_return)
                trial_return_history.append(trial_return)

            else:
                # Evaluate.
                eval_trial_return = 0.
                eval_trial_steps = 0
                if evaluation is not None:
                    env.set_test(test=True)
                    eval_obs = env.reset()
                    agent.reset()
                    eval_done = 0
                    while eval_done == 0:
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            env.render()
                        eval_trial_return += eval_r
                        eval_trial_steps += 1
                    # Episode done.
                    eval_trial_return_history.append(eval_trial_return)

                # Log stats.
                duration = time.time() - start_time
                combined_stats = {}
                if memory.nb_entries > 0:
                    # Print only if learing was happaning
                    stats = agent.get_stats()
                    for key in sorted(stats.keys()):
                        combined_stats[key] = mpi_mean(stats[key])

                    # Rollout statistics.
                    combined_stats['rollout/Q_mean'] = mpi_mean(qs)
                    combined_stats['rollout/actions_mean'] = mpi_mean(actions)
                    combined_stats['rollout/actions_std'] = mpi_std(actions)
                    combined_stats['rollout/trial_steps'] = mpi_mean(trial_steps)
                    combined_stats['rollout/return'] = mpi_mean(trial_returns)
                    combined_stats['rollout/return_history'] = mpi_mean(trial_return_history)

                    # Train statistics.
                    combined_stats['train/loss_actor'] = mpi_mean(train_actor_losses)
                    combined_stats['train/loss_critic'] = mpi_mean(train_critic_losses)
                    combined_stats['train/param_noise_distance'] = mpi_mean(train_adaptive_distances)

                # Evaluation statistics.
                if evaluation is not None:
                    combined_stats['eval/Q'] = mpi_mean(eval_q)
                    combined_stats['eval/return'] = eval_trial_return
                    combined_stats['eval/return_history'] = mpi_mean(eval_trial_return_history)
                    combined_stats['eval/steps'] = eval_trial_steps

                # Total statistics.
                combined_stats['total/duration'] = mpi_mean(duration)
                combined_stats['total/steps_per_second'] = mpi_mean(float(ts) / float(duration))
                combined_stats['total/trials'] = trial
                combined_stats['total/steps'] = ts

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                if rank == 0 and logdir:
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)
                    if evaluation and hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)

                # Reset statistics.
                trial_returns = []
                trial_steps = []
                actions = []
                qs = []
                train_actor_losses = []
                train_critic_losses = []
                train_adaptive_distances = []
                # End of evaluate and log statistics

            # Check if this is the last trial
            trial += 1
            if nb_trials and trial >= nb_trials:
                break
            if num_timesteps and ts >= num_timesteps:
                break

        # Saving policy and value function
        if save and saver and output != '':
            saver.save(sess, './%s' % output)

