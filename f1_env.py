import gym
from gym import spaces
import numpy as np
import random

from gym.envs.registration import register
from gym.utils.env_checker import check_env

register(
    id='f1_env_lap_simul',
    entry_point='f1_env:LapSimulation',
    max_episode_steps=300,
)

def timedelta(tire_type, tire_age):
    if tire_type == 0:
        a = 6/90
        c = -1
    elif tire_type == 1:
        a = 5/380
        c = 0
    else:
        a = 6/1740
        c = 0.5

    return a * tire_age**2 - a * tire_age + c

class LapSimulation(gym.Env) :

    metadata = {"render_modes": ['None'], "render_fps": 1}

    def __init__(self, render_mode=None):

        self._laps_left = 70
        self._tire_age = 0
        self._tire_type = 0
        self._leader_board = []
        self._agent_position = 0
        self._agent_time = 0

        self._bot_1_info = [0,0,0] #[time,tire_age, tire_type]
        self._bot_2_info = [0,0,0]

        # Observe a vector [laps_left, tire_age, tire_type, current_position]
        self.observation_space = spaces.MultiDiscrete([71,71,3,3])

        self.action_space = spaces.Discrete(4)

    def _get_obs(self) :
        return np.array([self._laps_left, self._tire_age, self._tire_type, self._agent_position])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._laps_left = 70
        self._tire_age = 0
        self._tire_type = 0
        self._leader_board = []
        self._agent_position = 0
        self._agent_time = 0

        self._bot_1_info = [0,0,0] #[time,tire_age, tire_type]
        self._bot_2_info = [0,0,0]

        obs = self._get_obs()

        info = {}

        return obs, info

    def step(self, action):
        self._laps_left += -1

        #bot part
        if self._laps_left == 23 or self._laps_left == 47:
            curr_lap_time_bot_1 = np.random.normal(70+timedelta(self._bot_1_info[2],self._bot_1_info[1]),0.25) + np.random.normal(15,1)
            self._bot_1_info = [self._bot_1_info[0],0,2]
        else:
            curr_lap_time_bot_1 = np.random.normal(70+timedelta(self._bot_1_info[2],self._bot_1_info[1]),0.25)

        self._bot_1_info = [self._bot_1_info[0]+curr_lap_time_bot_1,self._bot_1_info[1]+1,self._bot_1_info[2]]

        pit_stop_roll = random.randint(0,70)
        if pit_stop_roll <= 2:
            tire_type_roll = random.randint(0,2)
            curr_lap_time_bot_2 = np.random.normal(70 + timedelta(self._bot_2_info[2], self._bot_2_info[1]),0.25) + np.random.normal(15, 1)
            self._bot_2_info = [self._bot_2_info[0], 0, tire_type_roll]
        else:
            curr_lap_time_bot_2 = np.random.normal(70 + timedelta(self._bot_2_info[2], self._bot_2_info[1]), 0.25)

        self._bot_2_info = [self._bot_2_info[0] + curr_lap_time_bot_2, self._bot_2_info[1] + 1, self._bot_2_info[2]]

        #agent part
        if self._laps_left != 0 :
            if action == 3 :
                curr_lap_time_agent = np.random.normal(70+timedelta(self._tire_type,self._tire_age),0.25)
                self._tire_age += 1

            else:
                curr_lap_time_agent = np.random.normal(70+timedelta(action,0),0.2) + np.random.normal(15,1)
                self._tire_age = 0
                self._tire_type = action

            self._agent_time += curr_lap_time_agent

            gap_bot_1 = - (self._agent_time - self._bot_1_info[0])
            gap_bot_2 = - (self._agent_time - self._bot_2_info[0])

            if gap_bot_1 > 0:
                if gap_bot_2 > 0:
                    reward = 1
                    self._agent_position = 0
                else:
                    reward = 0
                    self._agent_position = 1
            elif gap_bot_2 < 0:
                reward = -1
                self._agent_position = 2
            else:
                reward = 0
                self._agent_position = 1

            observation = self._get_obs()
            terminated = False
            #info = {'gap_to_bot_1': gap_bot_1, 'curr_lap_time_bot_1': curr_lap_time_bot_1, 'gap_to_bot_2': gap_bot_2, 'curr_lap_time_bot_2': curr_lap_time_bot_2, 'curr_lap_time_agent': curr_lap_time_agent}
            info = {}
            return observation, reward, terminated, False, info

        else:
            terminated = True
            observation = self._get_obs()
            if self._agent_position == 0: reward = 70
            elif self._agent_position == 1: reward = 0
            else: reward = -10

            info = {}

            return observation, reward, terminated, False, info




if __name__ == '__main__':
    env = gym.make('f1_env_lap_simul', render_mode = 'None')
    #env = LapSimulation()
    print("check env begin")
    check_env(env.unwrapped)
    print("check env end")\

    #print(gym.spaces.utils.flatdim(env))
    #print(gym.spaces.utils.flatten_space())

    print(env.observation_space.sample())
    env.reset()
    sum = 0
    for i in range(70) :
        if i < 15: rand_action = 3
        if i == 15: rand_action = 2
        else: rand_action = 3
        obs, reward, terminated, _, info = env.step(rand_action)
        sum += reward
        print(i,rand_action,obs, sum, reward, info)