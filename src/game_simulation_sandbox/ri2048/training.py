import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
import tf_agents
import tqdm

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import matplotlib.pyplot as pl

from . import environment

tf.compat.v1.enable_v2_behavior()


def make_agent():
    env_name = "CartPole-v0"
    num_iterations = 1000
    collect_episodes_per_iteration = 5
    replay_buffer_capacity = 2000

    fc_layer_params = (32, 32, 32, 4)

    learning_rate = 1e-3
    log_interval = 25
    num_eval_episodes = 10
    eval_interval = 5

    train_env = environment.make_tf_environment()
    eval_env = environment.make_tf_environment()

    actor_net = tf_agents.networks.Sequential(
        layers=[
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
            tf_agents.keras_layers.InnerReshape((16, 32), (16 * 32,)),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax),
        ],
        input_spec=train_env.observation_spec(),
    )

    # actor_net = actor_distribution_network.ActorDistributionNetwork(
    #     train_env.observation_spec(),
    #     train_env.action_spec(),
    #     fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        use_advantage_loss=False,
        train_step_counter=train_step_counter,
    )
    tf_agent.initialize()

    print(actor_net.losses)
    actor_net.summary()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in tqdm.tqdm(range(num_iterations)):
        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_env,
            tf_agent.collect_policy,
            collect_episodes_per_iteration,
            replay_buffer,
        )

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print("\nstep = {0}: loss = {1}".format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(
                eval_env, tf_agent.policy, num_eval_episodes
            )
            print("\nstep = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)

        steps = np.arange(0, len(returns)) * eval_interval
        pl.clf()
        pl.plot(steps, returns, marker="o")
        pl.ylabel("Average Return")
        pl.xlabel("Step")
        pl.savefig("training.pdf")
        pl.savefig("training.png", dpi=150)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(environment, policy, num_episodes, replay_buffer):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1
