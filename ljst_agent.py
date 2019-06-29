import numpy as np
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.agent_base import Agent

STATE_SIZE = (176,1)
CONTEXT_LEN = 4


class LJST_Agent(Agent):
    def __init__(self, algorithm, action_dim, station_num, max_scheduling_num):
        super(LJST_Agent, self).__init__(algorithm)

        self.exploration = 1.1
        
        self.global_step = 0
        self.update_target_steps = 10000 // 4
        self.station_num = station_num
        self.max_scheduling_num = max_scheduling_num
        self.action_dim = action_dim
        
    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, STATE_SIZE[0], STATE_SIZE[1]],
                dtype='float32')
            self.value = self.alg.define_predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, STATE_SIZE[0], STATE_SIZE[1]],
                dtype='float32')
            action = layers.data(name='act', shape=[-1,17])
#             action1 = layers.data(name='act1', shape=[8])
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs',
                shape=[CONTEXT_LEN, STATE_SIZE[0], STATE_SIZE[1]],
                dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.define_learn(obs, action, reward,next_obs,terminal)

    def sample(self, obs):
        sample = np.random.random()
        if sample < self.exploration:
            ed = st = np.zeros(self.station_num)
            st[np.random.randint(0,self.station_num)] = 1  
            ed[np.random.randint(0,self.station_num)] = 1
            num = np.random.randint(0,self.max_scheduling_num)
            act = np.concatenate([st,ed,[num]])
        else:
            if np.random.random() < 0.01:
                st = np.zeros(self.station_num)
                ed = np.zeros(self.station_num)
                st[np.random.randint(0,self.station_num)] = 1  
                ed[np.random.randint(0,self.station_num)] = 1
                num = np.random.randint(0,self.max_scheduling_num) *10
                act = np.concatenate([st,ed,[num]])
            else:
                st = np.zeros(self.station_num)
                ed = np.zeros(self.station_num)
                obs = np.expand_dims(obs, axis=0)
                pred_Q = self.fluid_executor.run(
                    self.pred_program,
                    feed={'obs': obs.astype('float32')},
                    fetch_list=[self.value])[0]
                pred_Q = np.squeeze(pred_Q, axis=0)
                print('predQ',pred_Q.shape)
    #                 act = np.argmax(pred_Q)
                st[np.argmax(pred_Q[0:8])] = 1
                ed[np.argmax(pred_Q[8:16])] = 1
                num = int(pred_Q[16]) * 10
                print('agent_num',num)
                act = np.concatenate([st,ed,[num]])
                print('agent_act',act)
        self.exploration = max(0.1, self.exploration - 1e-5)
        return act


    def learn(self, obs, act, reward, next_obs, terminal):
#         if self.global_step % self.update_target_steps == 0:
#             self.alg.sync_target(self.gpu_id)
        self.global_step += 1

#         act = np.expand_dims(act, -1)
#         reward = np.clip(reward, -1, 1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost
