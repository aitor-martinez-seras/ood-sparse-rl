from multiprocessing import Process, Pipe
import gym
import numpy as np

def worker(conn, env, env_dict):
    
    while True:
        cmd, action, seed = conn.recv()
        
        if cmd == "step":
            obs, reward, done, info = env.step(action)
            
            if done:
                # special case
                # TODO: Aqui se podria meter la logica de cambio de mapas, metiendo en el env_dict
                #   la info del mapa que se quiere usar y cambiando el env en funcion de eso.
                env_key = list(env_dict.keys())[0]
                if env_key == 'MiniGrid-NumpyMapFourRoomsPartialView-v0':
                    possible_envs = list(env_dict.values())[0]
                    selected_env = np.random.choice(possible_envs)
                    env = gym.make(env_key,numpyFile='numpyworldfiles/' + selected_env,max_steps=100)
                
                # reset
                if seed is not None:
                    env.seed(int(seed))
                    # print('Seed reset with id:', seed)
                obs = env.reset()

            info = env.agent_pos
            conn.send((obs, reward, done, info))
        
        elif cmd == "reset":
            # special case
            env_key = list(env_dict.keys())[0]
            if env_key == 'MiniGrid-NumpyMapFourRoomsPartialView-v0':
                possible_envs = list(env_dict.values())[0]
                selected_env = np.random.choice(possible_envs)
                env = gym.make(env_key,numpyFile='numpyworldfiles/' + selected_env,max_steps=100)
                
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, env_dict):
        assert len(envs) >= 1, "No environment given."
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, env_dict))
            p.daemon = True
            p.start()
            remote.close()
        
        # Initialize seeds for all environments
        self.current_seeds = [None] * len(envs)  # Default None, meaning random seed
        
    def reset(self):
        for local in self.locals:
            local.send(("reset", None, None))
        
        # concat of Env0 & Env1,Env2,Env3...
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions, seeds=None):
        # update current seed value
        self.current_seeds = seeds
        
        # For Env1,Env2,Env3...
        for local, action, seed in zip(self.locals, actions[1:], self.current_seeds[1:]):
            local.send(("step", action, seed))
        
        # Env0
        obs, reward, done, info = self.envs[0].step(actions[0])
        # info = self.envs[0].agent_pos
        
        if done:
            # set seed to that env
            if self.current_seeds[0] is not None:
                self.envs[0].seed(int(self.current_seeds[0])) # HAS TO BE A INTEGER!
           
            # reset with updated seed
            obs = self.envs[0].reset()
            # print('Seed reset with id:', self.current_seeds[0])
            
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self,mode):
        return self.envs[0].render(mode)
        # raise NotImplementedError
