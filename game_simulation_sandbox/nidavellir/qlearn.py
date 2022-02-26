import asyncio
import datetime
import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor

import click
import coloredlogs
import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
import tqdm
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from . import environment

logger = logging.getLogger()


@click.command()
@click.argument("name")
def main(name: str):
    coloredlogs.install()
    logger.info("Make environment …")
    train_env = environment.make_tf_environment()
    logger.info("Make agent …")
    tf_agent = make_tf_agent(train_env)
    logger.info("Make replay buffer …")
    replay_buffer = make_replay_buffer(tf_agent, train_env)
    collect_episodes_per_iteration = 5
    batch_size = 64
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    num_iterations = 100
    logger.info("Start training …")

    train_dir = os.path.join(
        "train", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " " + name
    )
    train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(buffer_size=collect_episodes_per_iteration),
        tf_metrics.MaxReturnMetric(buffer_size=collect_episodes_per_iteration),
        tf_metrics.MinReturnMetric(buffer_size=collect_episodes_per_iteration),
        tf_metrics.ChosenActionHistogram(buffer_size=collect_episodes_per_iteration),
    ]
    observers = [
        replay_buffer.add_batch,
    ] + train_metrics
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        tf_agent.collect_policy,
        observers,
        num_episodes=collect_episodes_per_iteration,
    )
    # Initial driver.run will reset the environment and initialize the policy.
    for iteration in tqdm.tqdm(range(num_iterations)):
        final_time_step, policy_state = driver.run()

        for train_metric in train_metrics:
            train_metric.tf_summaries(
                train_step=iteration, step_metrics=train_metrics[:2]
            )

        experience, unused_info = next(iterator)
        trained = tf_agent.train(experience)


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
    train_step_counter = tf.Variable(0, dtype=tf.int64)
    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        summarize_grads_and_vars=True,
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


if __name__ == "__main__":
    main()
