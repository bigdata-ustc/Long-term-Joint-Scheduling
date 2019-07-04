import argparse
# import gym
import paddle.fluid as fluid
import numpy as np
import os
from ljst_agent import LJST_Agent
from ljst_model import LJST_Model
from ljst_env import Env
from collections import deque
from datetime import datetime
from replay_memory import ReplayMemory, Experience
from parl.algorithms import DQN
from parl.utils import logger
from tqdm import tqdm
# from utils import get_player

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 50
STATE_SIZE = (16,11)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 1e-4 * 0.5


def run_train_episode(env, agent, rpm):
    total_reward = 0
    all_cost = []
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        context = rpm.recent_state()
        context.append(state)
        context = np.stack(context, axis=0)
        action = agent.sample(context)
        next_state, reward, isOver, _ = env.step(action)
        logger.info('action:{} reward:{}'.format(action,reward))
        rpm.append(Experience(state, action, reward, isOver))
        # start training
        if rpm.size() > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = rpm.sample_batch(
                    args.batch_size)
                batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
                batch_next_state = batch_all_state[:, 1:, :, :]
                cost = agent.learn(batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_isOver)
                all_cost.append(float(cost))
        total_reward = reward
        state = next_state
        if isOver:
            logger.info('episode end total_reward:{}'.format(total_reward))
            break
#             
    if all_cost:
        logger.info('[Train]total_reward: {}, mean_cost: {}'.format(
            total_reward, np.mean(all_cost)))
    return total_reward, steps


def main():
    env = Env()
    station_num = env.station_num
    max_scheduling_num = env.max_scheduling_num
    rpm = ReplayMemory(MEMORY_SIZE, STATE_SIZE, CONTEXT_LEN)
#     action_num = 3
    action_dim = station_num
    act_dims = station_num * 2 + 1
    hyperparas = {
        'action_dim': act_dims,
        'lr': LEARNING_RATE,
        'gamma': GAMMA
    }
    model = LJST_Model(action_dim)
    algorithm = DQN(model, hyperparas)
    agent = LJST_Agent(algorithm, act_dims, station_num, max_scheduling_num)

    with tqdm(total=MEMORY_WARMUP_SIZE) as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            total_reward, steps = run_train_episode(env, agent, rpm)
            pbar.update(steps)

    # train
    pbar = tqdm(total=args.train_total_steps)
    recent_100_reward = []
    total_steps = 0
    max_reward = None
    while total_steps < args.train_total_steps:
        # start epoch
        total_reward, steps = run_train_episode(env, agent, rpm)
        total_steps += steps
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        pbar.update(steps)


    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e6),
        help='maximum environmental steps of games')
    args = parser.parse_args()

    main()
