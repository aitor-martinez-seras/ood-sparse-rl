import gym
import gym_minigrid
import numpy as np


def make_env(env_dict, seed=None):
    print(env_dict)
    env_key = list(env_dict.keys())[0]
    print('env_key:',env_key)

    if env_key == 'MiniGrid-NumpyMapMultiRoomPartialView-v0':
        possible_envs = list(env_dict.values())[0]
        selected_env = np.random.choice(possible_envs)
        filepath = 'numpyworldfiles/' + selected_env
        env = gym.make(env_key,numpyFile=filepath,max_steps=100)
        print('FILEPATH:',filepath)
        # print('env_values:',possible_envs)
        # print('selected env:',selected_env)
    else:
        env = gym.make(env_key)
        env.seed(seed)
    return env
