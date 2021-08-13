import asyncio
import logging
from concurrent.futures.thread import ThreadPoolExecutor

import click
import coloredlogs
import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
import tqdm
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from . import environment

logger = logging.getLogger()


@click.command()
def main():
    coloredlogs.install()
    logger.info("Make environment …")
    train_env = environment.make_tf_environment()
    logger.info("Make agent …")
    tf_agent = make_tf_agent(train_env)
    logger.info("Make replay buffer …")
    replay_buffer = make_replay_buffer(tf_agent, train_env)
    collect_episodes_per_iteration = 1
    batch_size = 128
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)
    logger.info("Collect initial episodes …")
    collect_episode(
        train_env,
        tf_agent.collect_policy,
        collect_episodes_per_iteration + batch_size,
        replay_buffer,
    )

    returns = []
    losses = []
    weights = []
    num_iterations = 10000
    eval_interval = 50
    num_eval_episodes = 100
    logger.info("Start training …")
    for _ in tqdm.tqdm(range(num_iterations)):
        collect_episode(
            train_env,
            tf_agent.collect_policy,
            collect_episodes_per_iteration,
            replay_buffer,
        )
        experience, unused_info = next(iterator)
        losses.append(tf_agent.train(experience).loss)
        step = tf_agent.train_step_counter.numpy()

        if step % eval_interval == 0:
            avg_return = compute_avg_return(
                train_env, tf_agent.policy, num_eval_episodes
            )
            returns.append(avg_return)
            weights.append(tf_agent._q_network.layers[0].get_weights())

            plot_returns(eval_interval, returns)
            plot_loss(losses)


def make_replay_buffer(tf_agent, train_env):
    replay_buffer_capacity = 100
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )
    return replay_buffer


def make_tf_agent(train_env):
    dense_layers = [
        tf.keras.layers.Embedding(26, 30, input_length=4),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6),
    ]
    q_net = sequential.Sequential(dense_layers)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )
    tf_agent.initialize()
    summarize_network(q_net)
    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)
    return tf_agent


def plot_returns(eval_interval, returns):
    steps = np.arange(0, len(returns)) * eval_interval
    pl.clf()
    pl.gcf().set_size_inches((7, 7))
    pl.plot(steps, returns, marker="o")
    pl.ylabel("Average Return")
    pl.xlabel("Step")
    pl.tight_layout()
    pl.savefig("training.pdf")
    pl.savefig("training.png", dpi=150)


def plot_loss(losses):
    steps = np.arange(1, len(losses) + 1)
    pl.clf()
    pl.gcf().set_size_inches((7, 7))
    pl.semilogy(steps, losses, marker=".", linestyle="none", alpha=0.5)
    pl.ylabel("Loss")
    pl.xlabel("Steps")
    pl.tight_layout()
    pl.savefig("loss.pdf")
    pl.savefig("loss.png", dpi=150)


def summarize_network(q_net):
    q_net.summary()
    for layer in q_net.layers:
        print(layer)
        print(layer.name)
        if hasattr(layer, "activation"):
            print(layer.activation)
        if hasattr(layer, "layers"):
            for layer in layer.layers[1:]:
                print(layer.name)
                print(layer.activation)


def compute_avg_return(env, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_episode(env, policy, num_episodes, replay_buffer):
    episode_counter = 0
    env.reset()

    while episode_counter < num_episodes:
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1
