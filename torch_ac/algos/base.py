from abc import ABC, abstractmethod
import time
import torch
import numpy as np
from copy import deepcopy
from collections import deque, OrderedDict
from statistics import mean

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch_ac.utils import RunningMeanStd,RewardForwardFilter,yvalue_richard_curve
from torch_ac.utils.intrinsic_motivation import CountModule,RNDModule,ICMModule,RIDEModule
from torch_ac.utils.im_models import EmbeddingNetwork_RAPID, EmbeddingNetwork_RIDE, \
                                        InverseDynamicsNetwork_RAPID, InverseDynamicsNetwork_RIDE, \
                                        ForwardDynamicsNetwork_RAPID, ForwardDynamicsNetwork_RIDE
from gym_minigrid.minigrid import Quicksand

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 separated_networks, env_name, num_actions,
                 int_coef, normalize_int_rewards,
                 im_type, int_coef_type,use_episodic_counts,use_only_not_visited,
                 total_num_frames,reduced_im_networks,
                 max_num_train_seeds,init_train_seed,
                 max_num_test_seeds,init_test_seed, envs_ood, env_ood_name, ood_method,
                 init_train_seed_ood, max_num_train_seeds_ood,
                 init_test_seed_ood, max_num_test_seeds_ood, ood_sampling_ratio,
                 intervene_after, intervene_every, adaptation_strategy,
                 window_visible):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module or tuple of torch.Module(s)
            the model(s); the separated_actor_critic parameter defines that
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        separated_networks: boolean
            set whether we are going to use a single AC neural network or
            two differents
        """

        # ENVIRONMENTS
        self.env = ParallelEnv(envs, env_name)
        # OOD envs. Only working in one proc for the moment
        if len(envs_ood) > 0:
            self.env_ood = ParallelEnv(envs_ood, env_ood_name)  # Only 1 proc
            self.ev_env_ood = ParallelEnv([envs_ood[0]], env_ood_name)  # Only 1 proc
        else:
            self.env_ood = None
            self.ev_env_ood = None
        # Evaluation env for the in_distr env
        self.ev_env = ParallelEnv([envs[0]], env_name) # just for evaluation

        # Store parameters

        self.separated_actor_critic = separated_networks
        self.reduced_im_networks = reduced_im_networks
        self.use_recurrence = True if recurrence > 1 else False
        self.acmodel = acmodel
        
        self.num_actions = num_actions
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.use_recurrence or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        if self.separated_actor_critic:
            for i in range(len(self.acmodel)):
                self.acmodel[i].to(self.device)
                self.acmodel[i].train()
        else:
            self.acmodel.to(self.device)
            self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.total_num_frames = total_num_frames
        print('total frames:',total_num_frames)
        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.agent_position = [None]*(shape[0])

        if self.use_recurrence:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values_ext = torch.zeros(*shape, device=self.device)
        self.advantages_ext = torch.zeros(*shape, device=self.device)
        self.returns_ext = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Level selection params
        self.max_num_train_seeds = max_num_train_seeds
        self.init_train_seed = init_train_seed
        self.level_ids = np.arange(self.max_num_train_seeds) + self.init_train_seed  # Generate the IDS, this can be modified on the fly to change levels
        self.specify_seeds_train = True if self.max_num_train_seeds > 0 else False
        self.current_eps_seeds = np.ones(shape[1], dtype=int) * self.init_train_seed # current episode seed
        self.next_eps_seeds = np.ones(shape[1], dtype=int) # next episodes seed
        # for evaluation environments too 
        self.max_num_test_seeds = max_num_test_seeds
        self.init_test_seed = init_test_seed

        # OOD
        self.init_train_seed_ood = init_train_seed_ood
        self.max_num_train_seeds_ood = max_num_train_seeds_ood
        self.init_test_seed_ood = init_test_seed_ood
        self.max_num_test_seeds_ood = max_num_test_seeds_ood
        self.ood_sampling_ratio = ood_sampling_ratio
        self.specify_seeds_train_ood = True if self.max_num_train_seeds_ood > 0 else False
        self.level_ids_ood = np.arange(self.max_num_train_seeds_ood) + self.init_train_seed_ood
        self.current_eps_seeds_ood = np.ones(shape[1], dtype=int) * self.init_train_seed_ood # current episode seed
        self.next_eps_seeds_ood = np.ones(shape[1], dtype=int) * self.init_train_seed_ood # next episodes seed
        self.current_level_is_ood = False  # Determines if the level is OOD or not
        self.obs_next_eps_ood = self.env_ood.reset() if self.env_ood else None
        self.obs_next_eps_in_distr = deepcopy(self.obs) if self.env_ood else None # stores the in_distr observations (for each proc)
        self.level_counter = 0  # Counts the number of levels that have been sampled randomly. It will reset everytime we intentionally sample an ood.
        self.num_levels_to_start_intervening = intervene_after  # Number of levels to sample randomly before starting to intervene and sample some manually
        self.manual_sample_every = intervene_every  # Sample one manually after completing X levels normally
        # Dicts that stores the OOD score of each detected level. Only the detected levels are stored. Detection depends on the method used.
        self.ood_score_of_levels = OrderedDict()  # {<id>_<true_distribution>: score}. Example: {15_ood: 1, 42_ind: 5}
        self.adaptation_strategy = adaptation_strategy  # Choose between 'online', 'offline'. Default: ''
        self.ood_module = None  # Stores the OOD module, if any. Regret and performance methods do not use it.
        self.method_type = ood_method if ood_method in ['', 'regret', 'performance'] else self.load_ood_method(ood_method)
        # if 'Quicksand' in list(env_name.keys())[0] or 'Quicksand' in list(env_ood_name.keys())[0]:
        #     self.quicksand_env = True
        #     self.quicksand_retention_steps = 5
        # else:
        #     self.quicksand_env = False
        # self.retention_counter = 0  # For the quicksand env
        # self.has_left_the_quicksand = True  # For the quicksand env

        # Initialize LOGs values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device) # monitores the return inside the episode (it increases with each step until done is reached)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs # monitores the total return that was given in the whole episode (updates after each episode)
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        self.episode_counter = 0
        self.frames_counter = 0

        # for intrinsic coef adaptive decay
        self.log_rollout_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_train = torch.tensor([],device=self.device) # stores the avg return reported after each rollout (avg of all penv after every nsteps)

        # *****  Intrinsic motivation related parameters *****
        self.int_coef = int_coef
        self.use_normalization_intrinsic_rewards = normalize_int_rewards


        self.im_type = im_type # module type ['rnd', 'icm', 'counts']
        self.int_coef_type = int_coef_type # ['static','parametric','adaptive']
        # define IM module
        if self.im_type == 'counts':
            print('\nUsing COUNTS')
            self.im_module = CountModule()

        elif self.im_type == 'rnd':
            print('\nUsing RND')
            if self.reduced_im_networks:
                self.im_module = RNDModule(neural_network_predictor=EmbeddingNetwork_RAPID(),
                                      neural_network_target=EmbeddingNetwork_RAPID(),
                                      device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RAPID().parameters())
                total_params = embedding_params*2
                print('***PARAMS with RAPID ARCHITECTURE:\nEmbedding {}\nTotal {}\n'.format(embedding_params,total_params))
            else:
                self.im_module = RNDModule(neural_network_predictor=EmbeddingNetwork_RIDE(),
                                      neural_network_target=EmbeddingNetwork_RIDE(),
                                      device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RIDE().parameters())
                total_params = embedding_params*2
                print('***PARAMS with RIDE ARCHITECTURE:\nEmbedding {}\nTotal {}\n'.format(embedding_params,total_params))
        elif self.im_type == 'icm':
            print('\nUsing ICM')
            if self.reduced_im_networks:
                self.im_module = ICMModule(emb_network = EmbeddingNetwork_RAPID(),
                                           inv_network = InverseDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           device = device)
            else:
                self.im_module = ICMModule(emb_network = EmbeddingNetwork_RIDE(),
                                           inv_network = InverseDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           device = device)

        elif self.im_type == 'ride':
            print('\nUsing RIDE')
            if self.reduced_im_networks:
                self.im_module = RIDEModule(emb_network = EmbeddingNetwork_RAPID(),
                                           inv_network = InverseDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RAPID().parameters())
                inv_dynamics_params = sum(p.numel() for p in InverseDynamicsNetwork_RAPID(num_actions=self.num_actions).parameters())
                forw_dynamics_params = sum(p.numel() for p in ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions).parameters())
                total_params = embedding_params + inv_dynamics_params + forw_dynamics_params
                print('***PARAMS with RAPID ARCHITECTURE:\nEmbedding {}\nInverse {}\nForward {}\nTotal {}\n'.format(embedding_params,inv_dynamics_params,forw_dynamics_params,total_params))
            else:
                self.im_module = RIDEModule(emb_network = EmbeddingNetwork_RIDE(),
                                           inv_network = InverseDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RIDE().parameters())
                inv_dynamics_params = sum(p.numel() for p in InverseDynamicsNetwork_RIDE(num_actions=self.num_actions).parameters())
                forw_dynamics_params = sum(p.numel() for p in ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions).parameters())
                total_params = embedding_params + inv_dynamics_params + forw_dynamics_params
                print('***PARAMS with RIDE ARCHITECTURE:\nEmbedding {}\nInverse {}\nForward {}\nTotal {}\n'.format(embedding_params,inv_dynamics_params,forw_dynamics_params,total_params))

        # episodic counts and first visit variables
        self.use_episodic_counts = 1 if im_type == 'ride' else use_episodic_counts # ride always uses episodic counts by default
        self.episodic_counts = [CountModule() for _ in range(self.num_procs)] # counts used to carry out how many times each observation has been visited inside an episode
        self.use_only_not_visited = use_only_not_visited
        self.visited_state_in_episode = torch.zeros(*shape, device=self.device) # mask that is used to allow or not compute a non-zero intrinsic reward

        # Parameters needed when using two-value/advantage combination for normalization
        self.return_rms = RunningMeanStd()
        self.normalization_int_score = 0
        self.min_std = 0.01
        self.predominance_ext_over_int = torch.zeros(*shape, device=self.device)

        # experience values
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.rewards_total = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)
        self.advantages_total = torch.zeros(*shape, device=self.device)
        self.returns_int = torch.zeros(*shape, device=self.device)
        # add monitorization for intrinsic part
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
        # other for normalization
        self.log_episode_return_int_normalized = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_normalized =  [0] * self.num_procs
        # add avg 100 episodes return
        self.last_100return = deque([0],maxlen=100)
        self.last_100success = deque([0],maxlen=100)
        self.ngu_episode_return = 0
        
        # To see the display Window
        self.window_visible = window_visible
        self.first_step_of_env = True
        if self.window_visible:
            from gym_minigrid.window import Window
            self.window = Window('gym_minigrid - ')
            # self.env.render('rgb_array')
            # if self.env_ood:
            #     self.env_ood.render('rgb_array')
            # self.env.render('human')
            # self.env_ood.render('human')

        # Reset envs with the first seeds (if specified) and define next seeds
        # Always start with in_distr env in self.obs
        if self.specify_seeds_train and self.env_ood:
            for idx_env in range(len(self.env.envs)):
                self.next_eps_seeds[idx_env] = self.select_next_seed(level_ids=self.level_ids, strategy='specific_ids_uniform')
                self.env.envs[idx_env].seed(int(self.current_eps_seeds[idx_env]))
                self.obs_next_eps_in_distr[idx_env] = self.env.envs[idx_env].reset()
                self.obs[idx_env] = deepcopy(self.obs_next_eps_in_distr[idx_env])  # We start with a in_distr level
        # Reset ood envs with the first seeds (if specified) and define next seeds
        if self.specify_seeds_train_ood and self.env_ood:
            for idx_env in range(len(self.env_ood.envs)):
                self.next_eps_seeds_ood[idx_env] = self.select_next_seed(level_ids=self.level_ids_ood, strategy='specific_ids_uniform')
                self.env_ood.envs[idx_env].seed(int(self.current_eps_seeds_ood[idx_env]))
                self.obs_next_eps_ood[idx_env] = self.env_ood.envs[idx_env].reset()

        print('num_frame per proc:',self.num_frames_per_proc)
        print('num of process:',self.num_procs)
        print('num frames (num_pallel envs*framesperproc):', self.num_frames)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        forw_time_total = []
        for i in range(self.num_frames_per_proc):

            # update frame counter after each step
            self.frames_counter += self.num_procs

            if self.int_coef_type == 'ngu':
                beta_values = [yvalue_richard_curve(im_coef=self.int_coef,im_type=self.im_type,max_steps=self.num_procs,timestep=i) for i in range(self.num_procs)]
                actual_int_coef = torch.tensor(beta_values,device=self.device).float()
                actual_int_coef[-1] = 0
                # print('int coef',actual_int_coef)
                # input()

            # Do one agent-environment interaction           
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            
            # to save window visualization
            if self.window_visible and self.first_step_of_env:
                if self.current_level_is_ood:
                    in_distr_or_ood = 'OOD'
                    _env = self.env_ood
                    cur_seed = self.current_eps_seeds_ood[0]
                else:
                    in_distr_or_ood = 'NORMAL'
                    _env = self.env
                    cur_seed = self.current_eps_seeds[0]
                # img = _env.render('rgb_array')
                # self.window.show_img(img)
                # self.window.set_caption(f'{in_distr_or_ood} Level seed: {cur_seed:02d}, Episode: {self.episode_counter}, step: {i}')
                # self.window.fig.savefig(f'ood_storage/figs/{in_distr_or_ood}_seed_{cur_seed:02d}_{i}.png')
                # self.first_step_of_env = False
                #_env.render('human')

            with torch.no_grad():
                if self.use_recurrence:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    if self.separated_actor_critic:
                        dist = self.acmodel[0](preprocessed_obs)
                        value = self.acmodel[1](preprocessed_obs)
                    else:
                        if self.int_coef_type == 'ngu':
                            dist, value = self.acmodel(obs=preprocessed_obs,int_coefs=actual_int_coef)
                        else:
                            dist, value = self.acmodel(obs=preprocessed_obs)

            # take action from distribution
            action = dist.sample()

            # In quicksand env, the agent must be retained in the quicksand for some steps
            # if self.quicksand_env:
            #     if self.has_left_the_quicksand:  # If the agent has left the quicksand, it can be blocked again
            #         if isinstance(self.env_ood.envs[0].env.grid.get(*self.env_ood.envs[0].env.agent_pos), Quicksand):  # Working only with balls for the moment
            #             self.retention_counter += 1
            #             if self.retention_counter < self.quicksand_retention_steps:
            #                 action = torch.tensor([6], device=self.device, dtype=torch.int)
            #             else:  # Time to unblock the agent in the quicksand
            #                 self.retention_counter = 0
            #                 self.has_left_the_quicksand = False
            #     else:
            #         if not isinstance(self.env_ood.envs[0].env.grid.get(*self.env_ood.envs[0].env.agent_pos), Quicksand):
            #             self.has_left_the_quicksand = True                    
            
            # *** ENVIRONMENT STEP ***

            # if self.specify_seeds:
            #     obs, reward, done, agent_pos = _env.step(actions=action.cpu().numpy(),seeds=self.next_eps_seeds)
            # else:
            #     obs, reward, done, agent_pos = _env.step(actions=action.cpu().numpy(),seeds=len(self.next_eps_seeds)*[None])
            
            if self.current_level_is_ood:
                if self.specify_seeds_train_ood:
                    obs, reward, done, agent_pos = self.env_ood.step(actions=action.cpu().numpy(),seeds=self.next_eps_seeds_ood)
                else:
                    obs, reward, done, agent_pos = self.env_ood.step(actions=action.cpu().numpy(),seeds=len(self.next_eps_seeds_ood)*[None])
            else:
                if self.specify_seeds_train:
                    obs, reward, done, agent_pos = self.env.step(actions=action.cpu().numpy(),seeds=self.next_eps_seeds)
                else:
                    obs, reward, done, agent_pos = self.env.step(actions=action.cpu().numpy(),seeds=len(self.next_eps_seeds)*[None])

            # Update experiences values

            self.obss[i] = self.obs # stores the current observation on the experience
            self.obs = obs # stores the next_obs obtained after the step in the env
            self.agent_position[i] = agent_pos

            if self.use_recurrence:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values_ext[i] = value

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # ***Intrinsic motivation - calculate intrinsic rewards
            if self.int_coef > 0:

                # calculate bonus (step by step)  - shape [num_procs, 7,7,3]
                input_current_obs = preprocessed_obs
                input_next_obs = self.preprocess_obss(self.obs, device=self.device) # contains next_observations

                # FOR COMPUTING INTRINSIC REWARD, THE REQUIRED SHAPE IS JUST A UNIT -- i.e image of [7,7,3]; action of [1] (it is calculated one by one)
                # FOR UPDATING COUNTS (done IN BATCH for efficiency), the shape requires to have the batch-- i.e image of [batch,7,7,3]; action of [batch,1]

                forw_time_i = time.time()
                rewards_int = [self.im_module.compute_intrinsic_reward(obs=ob,next_obs=nobs,coordinates=coords,actions=act) \
                                            for ob,nobs,coords,act in zip(input_current_obs.image, input_next_obs.image, agent_pos, action)]
                forw_time_f = time.time()
                forw_time_total.append(forw_time_f - forw_time_i)

                self.rewards_int[i] = rewards_int_torch = torch.tensor(rewards_int,device=self.device,dtype=torch.float)


                # Reward only when agent visits state s for the first time in the episode
                if self.use_only_not_visited:

                    # check if state has already been visited -- mask
                    self.visited_state_in_episode[i] = torch.tensor([self.episodic_counts[penv].check_ifnot_already_visited(next_obs=nobs,actions=act) \
                                                        for penv,(ob,nobs,coords,act) in enumerate(zip(input_current_obs.image, input_next_obs.image, agent_pos, action))])

                    self.rewards_int[i] = rewards_int_torch * self.visited_state_in_episode[i]

                # To use episodic counts (mandatory in the case of ride)
                if self.use_episodic_counts or self.use_only_not_visited:
                    # ***UPDATE EPISODIC COUNTER (mandatory for both episodic and 1st visitation count strategies)***
                    current_episode_count_reward = np.array([self.episodic_counts[penv].update(obs=ob,next_obs=nobs,coordinates=coords,actions=act) \
                                                    for penv,(ob,nobs,coords,act) in enumerate(zip(input_current_obs.image.unsqueeze(1), input_next_obs.image.unsqueeze(1) , agent_pos, action.unsqueeze(1).unsqueeze(1) ) )]) # we need to squeeze to have actions of shape [num_procs, 1, 1] and also the observations [num_procs,1,7,7,3]
                    # the update function returns the sqrt inverse value of counts (the intrinsic reward value in counts for the next_state)
                    if self.use_episodic_counts:
                    # Divide/multiply by episodic counts

                        current_episode_count_reward = torch.from_numpy(current_episode_count_reward).to(self.device)
                        self.rewards_int[i] = rewards_int_torch * current_episode_count_reward.squeeze(1)
                        # print('Actual rewards:',rewards_int_torch)
                        # print('Current episode counter:',current_episode_count_reward)
                        # print('Final Rewards:', self.rewards_int[i])

            # Update log values
            if self.int_coef > 0:
                self.log_episode_return_int += rewards_int_torch
                self.log_episode_return_int_normalized += torch.tensor(np.asarray(rewards_int)/max(self.normalization_int_score,self.min_std), device=self.device, dtype=torch.float)
                # used for adaptive int coef
                self.log_rollout_return_int += rewards_int_torch
                # print('log episode_return:',self.log_episode_return_int)
                # print('log rollout return',self.log_rollout_return_int)

            if self.int_coef_type == 'ngu':
                # we just take into account the last agent, which has beta=0
                self.ngu_episode_return += torch.tensor(reward[-1], device=self.device, dtype=torch.float)
                ngu_mask = self.mask[-1]

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)


            # for all the num_procs...
            for i, done_ in enumerate(done):
                if done_:
                    self.episodic_counts[i].reset()
                    
                    ######################################################
                    # Level selection process
                    ######################################################

                    self.first_step_of_env = True  # For the window visualization

                    # Decide if the level must be stored for future training or not
                    if self.env_ood and self.adaptation_strategy == 'online':
                        if self.method_type == "ood":
                            pass
                        elif self.method_type == "regret":
                            pass
                        elif self.method_type == "performance":
                            if reward[i] < 0.0001:
                                if self.current_level_is_ood:
                                    # Store the OOD level
                                    self.ood_score_of_levels[str(self.current_eps_seeds_ood[i]) + '_ood'] = 1
                                else:
                                    # Store the in_distr level
                                    self.ood_score_of_levels[str(self.current_eps_seeds[i]) + '_ind'] = 1
                        else:
                            raise ValueError("The method type is not valid. Please choose between ood, regret or performance.")
                            
                    # Select the seed of the next level
                    if self.current_level_is_ood:
                        # reset seed of the OOD env as the level finished is OOD
                        if self.specify_seeds_train_ood:
                            self.current_eps_seeds_ood[i] = self.next_eps_seeds_ood[i]
                            self.next_eps_seeds_ood[i] = self.select_next_seed(level_ids=self.level_ids_ood, strategy='specific_ids_uniform')
                            #self.seeds_ood[i] = self.select_next_seed(strategy='uniform_range')
                    else:
                        # reset seed of the in_distr env as the level finished is in_distr 
                        if self.specify_seeds_train:
                            self.current_eps_seeds[i] = self.next_eps_seeds[i]
                            self.next_eps_seeds[i] = self.select_next_seed(level_ids=self.level_ids, strategy='specific_ids_uniform')
                            #self.next_eps_seeds[i] = self.select_next_seed(strategy='uniform_range')
                            # print('\nProc{} -- Seed that has already been applied: {}'.format(i,self.seed[i]))
                            # print('Seed selected for the next episode/level {}'.format(self.next_eps_seeds[i]))

                    # TODO: Only working with one proc
                    # Select if the next level is OOD or in_distr
                    if self.env_ood:
                        self.level_counter += 1

                        # Manual decision of the next level being OOD or in_distr. By default, it is deactivated. This is assured by the assertions in train.py, 
                        #   where the intervene_after and intervene_every parameters are forced to be -1 if no adaptation strategy is selected. Therefore, as 
                        #   the default value of intervene_every is -1, the manual selection is deactivated by  the check self.num_levels_to_start_intervening >= 0.
                        
                        _diff = self.level_counter - self.num_levels_to_start_intervening
                        if (_diff % self.manual_sample_every == 0) and (_diff >= 0) and (self.num_levels_to_start_intervening >= 0):
                            # Sample one manually after completing X levels normally and only after completing the first X levels 
                            #   and only if the number of levels to start intervening is greater than 0 as default is -1 (deactivated)
                            
                            # Select the next seed from the level ids
                            possible_levels = list(self.ood_score_of_levels.keys())
                            # Check if there are levels selected yet
                            if len (possible_levels) == 0:
                                raise ValueError("The are no levels selected yet. MANUALLY CHANGE THIS CODE")
                            
                            if self.method_type == "ood":
                                probs = list(self.ood_score_of_levels.values())
                                selected_level = self.select_next_seed(level_ids=possible_levels, strategy="specific_ids_weighted", probabilities=probs)
                            # Each method has to have each own way of assigning probabilities
                            elif self.method_type == "regret":
                                pass
                            elif self.method_type == "performance":
                                selected_level = self.select_next_seed(level_ids=possible_levels, strategy="specific_ids_uniform")
                            else:
                                raise ValueError("The method type is not valid. Please choose between ood, regret or performance.")
                            
                            # Assing the seed of the new level and reset the env
                            id_new_level, true_environment = selected_level.split('_')
                            id_new_level = int(id_new_level)
                            if true_environment == 'ind':  # The new level is not OOD
                                self.current_level_is_ood = False
                                self.next_eps_seeds[0] = self.current_eps_seeds[0]
                                self.current_eps_seeds[0] = id_new_level
                                self.env.envs[0].seed(int(id_new_level))
                                self.obs = (self.env.envs[0].reset(),)  # Only work with one proc                                
                            else:  # The new level is OOD
                                self.current_level_is_ood = True
                                self.next_eps_seeds_ood[0] = self.current_eps_seeds_ood[0]
                                self.current_eps_seeds_ood[0] = id_new_level
                                self.env_ood.envs[0].seed(int(id_new_level))
                                self.obs = (self.env_ood.envs[0].reset(),)  # Only work with one proc

                        # Decidision of the next level being OOD or in_distr randomly, when no adaptation strategy is selected and an OOD environment is used.
                        else:
                            # Store the obs of the next level (in_distr or OOD).
                            # It is necessary because the obs is reseted after episode in done, therefore if we change from in_distr to OOD, 
                            #   we need to store the obs of the in_distr level to be able to use it the next time we sample an in_distr level.
                            # In case the level does not change (from in_distr to OOD or viceversa), the self.obs just makes two deepcopy
                            #   operations of the same obs. This behaviour can be optimized, but the created overhead is not significant, for one proc at least.
                            if self.current_level_is_ood:
                                self.obs_next_eps_ood = deepcopy(self.obs)
                            else:
                                self.obs_next_eps_in_distr = deepcopy(self.obs)
                            # Sample next level (in_distr or OOD)
                            if np.random.rand() < self.ood_sampling_ratio:  # if 0.25, 1/4 of the time the random number will be < 0.25 and we will generate an OOD level
                                # The next level will be OOD
                                self.current_level_is_ood = True
                                self.obs = deepcopy(self.obs_next_eps_ood)
                            else:
                                # The next level will be in_distr
                                self.current_level_is_ood = False
                                self.obs = deepcopy(self.obs_next_eps_in_distr)
                    
                    ######################################################
                    # End of the level selection process
                    ######################################################
                          
                    # log related
                    if self.int_coef > 0:
                        self.log_return_int.append(self.log_episode_return_int[i].item())
                        self.log_return_int_normalized.append(self.log_episode_return_int_normalized[i].item())
                    self.log_done_counter += 1
                    self.episode_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    # save avg return of last 100 episodes
                    if self.int_coef_type == 'ngu' and (i == len(done) - 1):
                        self.last_100return.append(self.ngu_episode_return.item())
                        # if self.ngu_episode_return > 0:
                        # print('ngu score:',self.ngu_episode_return.item())
                    else:
                        self.last_100return.append(self.log_episode_return[i].item())
                        self.last_100success.append(1 if self.log_episode_return[i].item() > 0 else 0)

            # Intrinsic Return related
            if self.int_coef > 0:
                # resetea log a 0 si el episodio habia terminado (mask != done)
                self.log_episode_return_int *= self.mask
                self.log_episode_return_int_normalized *= self.mask

            if self.int_coef_type == 'ngu':
                self.ngu_episode_return *= ngu_mask

            self.log_episode_return *= self.mask #
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            # **********************************************************************
            # ONE STEP INSIDE THE ROLLOUT COMPLETED
            # **********************************************************************


        # **********************************************************************
        # ROLLOUT COLLECTION FINISHED.
        # **********************************************************************

        # 1.Update IM Module
        # 2.Normalize intrinsic rewards (before training)

        # Part 1 of updating...
        if self.int_coef > 0:
            # 1.1. preprocess the batch of data to be Tensors
            shape_im = (self.num_frames_per_proc,self.num_procs, 7,7,3) # preprocess batch observations (num_steps*num_instances, 7 x 7 x 3)
            input_obss = torch.zeros(*shape_im,device=self.device)
            input_nobss = torch.zeros(*shape_im,device=self.device)

            # generate next_states (same as self.obss + an additional next_state of al the penvs)
            nobss = deepcopy(self.obss)
            nobss = nobss[1:] # pop first element and move left
            nobss.append(self.obs) # add at the last position the next_states

            for num_frame,(mult_obs,mult_nobs) in enumerate(zip(self.obss,nobss)): # len(self.obss) ==> num_frames_per_proc == number_of_step

                for num_process,(obss,nobss) in enumerate(zip(mult_obs,mult_nobs)):
                    o = torch.tensor(obss['image'], device=self.device)
                    no = torch.tensor(nobss['image'], device=self.device)
                    input_obss[num_frame,num_process].copy_(o)
                    input_nobss[num_frame,num_process].copy_(no)

            # 1.2. reshape to have [num_frames*num_procs, 7, 7, 3]
            input_obss = input_obss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_nobss = input_nobss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_actions = self.actions.view(self.num_frames_per_proc*self.num_procs,-1)

            # self.im_module.visualize_counts()

            # 1.3. Update
            backw_time_i = time.time()
            self.im_module.update(obs=input_obss,next_obs=input_nobss,actions=input_actions,coordinates=self.agent_position)
            # Calculate times related to neural networks
            backw_time_f = time.time()
            backw_time_total = backw_time_f - backw_time_i
            # update forward time also
            forw_time_total = np.sum(forw_time_total)
            total_time_fwandbw = forw_time_total + backw_time_total

            # 2. Normalize (if required)
            if self.use_normalization_intrinsic_rewards:
                # Calculate normalization after each rollout
                batch_mean, batch_var, batch_count = self.log_rollout_return_int.mean(-1).item(), self.log_rollout_return_int.var(-1).item(), len(self.log_rollout_return_int)
                self.return_rms.update_from_moments(batch_mean, batch_var, batch_count)
                self.normalization_int_score = np.sqrt(self.return_rms.var)
                self.normalization_int_score = max(self.normalization_int_score, self.min_std)

                # apply normalization
                self.rewards_int /= self.normalization_int_score


        # obtain next_value for computing advantages and return
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        with torch.no_grad():
            if self.use_recurrence:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                if self.separated_actor_critic:
                    next_value = self.acmodel[1](preprocessed_obs)
                else:
                    if self.int_coef_type == 'ngu':
                        _, next_value = self.acmodel(obs=preprocessed_obs,int_coefs=actual_int_coef)
                    else:
                        _, next_value = self.acmodel(obs=preprocessed_obs)

        # **********************************************************************
        # Calculate new int coef(beta_t) based on static, parametric or adaptive decays
        # **********************************************************************
        hist_ret_avg = 1 #default value (to monitore)

        if self.int_coef_type == 'static':
            actual_int_coef = self.int_coef
        elif self.int_coef_type == 'parametric':
            actual_int_coef = yvalue_richard_curve(im_coef=self.int_coef,im_type=self.im_type,max_steps=self.total_num_frames,timestep=self.frames_counter)
        elif self.int_coef_type == 'adaptive' or self.int_coef_type == 'adaptive_1000':

            # update historic avg with the avg return collected by all the agents
            avg = self.log_rollout_return_int.mean(-1)
            self.log_return_int_train = torch.cat((self.log_return_int_train,avg.unsqueeze(0)), dim=0)

            # get historical return avg (same for all agents)
            if self.int_coef_type == 'adaptive_1000':
                hist_ret_avg = torch.mean(self.log_return_int_train[-1000:])
            else:
                hist_ret_avg = torch.mean(self.log_return_int_train)

            # one different actual_inf_coef for each agent
            decay_aux = torch.zeros(self.num_procs)
            for penv in range(self.num_procs):
                decay_aux[penv] = min(1, self.log_rollout_return_int[penv]/hist_ret_avg)

            actual_int_coef = self.int_coef*decay_aux
            hist_ret_avg = hist_ret_avg.item()

        # *** reinit as only has rollout scope
        self.log_rollout_return_int = torch.zeros(self.num_procs, device=self.device) # do it here, not anywhere else; adaptive method requires reset after calculation


        # ***Combining EXTRINSIC-INTRINSIC rewards***
        if self.int_coef <= 0:
            # No intrinsic_rewards
            self.rewards_total.copy_(self.rewards)

            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values_ext[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages_ext[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values_ext[i]
                self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            self.returns_ext =  self.values_ext + self.advantages_ext
            self.advantages_total.copy_(self.advantages_ext)

        # USING INTRINSIC MOTIVATION
        else:
            # 1. *** r_total = r_ext + Beta*r_int ***
            if self.int_coef_type == 'adaptive' or self.int_coef_type == 'adaptive_1000' or self.int_coef_type == 'ngu':
                for penv in range(self.num_procs):
                    self.rewards_total[:,penv] = self.rewards[:,penv] + actual_int_coef[penv]*self.rewards_int[:,penv]
                    self.rewards_total[:,penv] /= (1+actual_int_coef[penv])
            else:
                # more efficient (if non-adaptive used)
                self.rewards_total = self.rewards + actual_int_coef*self.rewards_int
                self.rewards_total /= (1+actual_int_coef)

            # 2. Calculate advantages and returns
            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values_ext[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages_ext[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values_ext[i]
                self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            self.returns_ext =  self.values_ext + self.advantages_ext
            self.advantages_total.copy_(self.advantages_ext)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.use_recurrence:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # extrinsic stream (used normally)
        exps.value_ext = self.values_ext.transpose(0, 1).reshape(-1)
        exps.advantage_ext = self.advantages_ext.transpose(0, 1).reshape(-1)
        exps.returnn_ext = self.returns_ext.transpose(0, 1).reshape(-1)

        # additional intrinsic stream required when using two-streams instead of one
        exps.advantage_int = self.advantages_int.transpose(0, 1).reshape(-1)
        exps.advantage_total = self.advantages_total.transpose(0,1).reshape(-1)
        exps.returnn_int = self.returns_int.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Add actual_int_coefs if used as input for Actor-Critic (for ngu case mandatory)
        shape = (self.num_frames_per_proc, self.num_procs)
        int_coefs = torch.zeros(*shape, device=self.device)
        for penv in range(self.num_procs):
            v = actual_int_coef[penv] if ((self.int_coef_type == 'ngu') or (self.int_coef_type == 'adaptive') or (self.int_coef_type == 'adaptive_1000')) else actual_int_coef
            int_coefs[:,penv] = v
        exps.int_coefs = int_coefs.transpose(0, 1).reshape(-1)

        ########################################################################
        # Log some values
        ########################################################################

        # weight of int coef to monitorize
        if self.int_coef_type=='static' or self.int_coef_type=='parametric':
            # only one value is assumed for all the penvs
            weight_int_coef = actual_int_coef
        elif self.int_coef_type=='adaptive' or self.int_coef_type=='adaptive_1000':
            # we store the avg among all the penvs weight coef
            weight_int_coef = actual_int_coef.mean(-1)
        else:
            # with ngu, take the agent with lowest value
            weight_int_coef = actual_int_coef[-1]


        keep = max(self.log_done_counter, self.num_procs)
        
        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "return_int_per_episode": self.log_return_int[-keep:],
            "return_int_per_episode_norm": self.log_return_int_normalized[-keep:],
            "normalization_int_score": self.normalization_int_score,
            "episode_counter": self.episode_counter,
            "avg_return": mean(self.last_100return),
            "avg_success": mean(self.last_100success),
            "weight_int_coef": weight_int_coef,
            "predominance_ext_over_int": self.predominance_ext_over_int.mean().item(),
            "hist_ret_avg":hist_ret_avg,
        }
        # If using INTRINSIC MOTIVATION, log some additional values
        if self.int_coef > 0:
            logs.update({"time_forw_and_backw":total_time_fwandbw})

        # OOD logs
        if self.env_ood:
            num_in_distr_levels = 0
            num_ood_levels = 0
            for lvl in self.ood_score_of_levels.keys():
                if lvl.split("_")[1] == "ood":
                    num_ood_levels += 1
                else:
                    num_in_distr_levels += 1
            dict_ood_levels = {
                "number_of_ood_levels_detected": num_ood_levels,
                "number_of_in_distr_levels_detected": num_in_distr_levels,
            }
            logs.update(dict_ood_levels)

        # sobreescribe para redimensionar y empezar una nueva colecta
        self.log_done_counter = 0
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_return_int_normalized = self.log_return_int_normalized[-self.num_procs:]
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def select_next_seed(self, strategy, level_ids=None, probabilities=None):
        
        if strategy == 'uniform_range':
            return np.random.randint(low=self.init_train_seed, high=self.init_train_seed + self.max_num_train_seeds)

        elif strategy == 'range_weighted':
            return np.random.choice(a=np.arange(self.init_train_seed,self.max_num_train_seeds), p=probabilities)

        elif strategy == 'specific_ids_uniform':
            return np.random.choice(a=level_ids)

        elif strategy == 'specific_ids_weighted':
            return np.random.choice(a=level_ids, p=probabilities)
        else:
            raise ValueError("Invalid seed selection strategy")

    # TODO: Only working with 1 proc currently
    def evaluate_agent(self, strategy='level_ids_in_order', eval_in_ood_envs=False):
        """
            Evaluate the agent into a number of levels
        """
        # TODO: Move this to the init function
        if eval_in_ood_envs:
            envs = self.ev_env_ood
            init_seed = self.init_test_seed_ood
            last_seed = self.init_test_seed_ood + self.max_num_test_seeds_ood
            seeds = self.current_eps_seeds_ood
        else:
            envs = self.ev_env
            init_seed = self.init_test_seed
            last_seed = self.init_test_seed + self.max_num_test_seeds
            seeds = self.current_eps_seeds

        if strategy == 'level_ids_in_order':
            level_ids = np.arange(init_seed, last_seed)

        num_levels = len(level_ids)
        average_success = np.zeros(num_levels)
        average_steps = np.zeros(num_levels)
        average_return = np.zeros(num_levels)
        ood_or_in_distr = 'OOD' if eval_in_ood_envs else 'Normal'
        print(f'{ood_or_in_distr}: Evaluating in {num_levels} levels')

        # # To see the display Window
        # window_visible = True
        # if window_visible:
        #     from gym_minigrid.window import Window
        #     from time import sleep
        #     window = Window('gym_minigrid - ')
        #     name_for_png = 'test_ood' if eval_in_ood_envs else 'test_in_distr'

        for e in range(num_levels):
            next_seed = level_ids[e]
            # Initialize environment with next_seed
            envs.envs[0].seed(int(next_seed))
            obs = [envs.envs[0].reset()]
            steps = 0

            while True:

                # # to see window (interactive)
                # if window_visible and steps==0:
                #     img = envs.render('rgb_array')
                #     window.show_img(img)
                #     window.set_caption(f'Level id: {next_seed}')
                #     window.fig.savefig(f'ood_storage/figs/a_step_0_{name_for_png}_id_{next_seed:03d}_{j}.png')

                steps += 1

                preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                with torch.no_grad():
                    if self.separated_actor_critic:
                        distribution = self.acmodel[0](preprocessed_obs)
                        value = self.acmodel[1](preprocessed_obs)
                    else:
                        distribution, value = self.acmodel(obs=preprocessed_obs)

                action = distribution.sample()
                #obs, reward, done, _ = envs.step(actions=action, seeds=[next_seed])
                obs, reward, done, _ = envs.step(actions=action, seeds=len(seeds)*[None])
                reward = np.array(reward)
                done = np.array(done)

                if done:
                    average_success[e] = 1 if reward > 0 else 0
                    average_steps[e] = steps
                    average_return[e] = reward
                    break

        return np.mean(average_success), np.mean(average_steps),np.mean(average_return)
    
    # REMOVE
    def create_pool_of_levels(self, ood_detection_performance_eval=False):
        """
            Create a pool of levels to be used during offline training
        """
        # To see the display Window
        window_visible = False
        
        num_levels_each_type = [len(self.level_ids), len(self.level_ids_ood)]
        seeds = self.current_eps_seeds

        # For counting OOD states, we have to now in advance the OOD elements introduced in the environment
        if ood_detection_performance_eval:
            ood_ids_list = []
            ood_colors_list = []
            ind_env_id = self.env.envs[0].spec.id.split('-')[1]
            ood_env_id = self.env_ood.envs[0].spec.id.split('-')[1]
            if ("Lava" in ood_env_id) or ("L" in ood_env_id):
                if not ("Lava" in ind_env_id):  # This is for the case when the in_distr env is Lava and the OOD env is Lava+Ball
                    ood_ids_list.append(9)
            if ("Ball" in ood_env_id) or ("B" in ood_env_id):
                if not ("Ball" in ind_env_id):  # This is for the case when the in_distr env is Ball and the OOD env is Lava+Ball
                    ood_ids_list.append(6)
            # In case the color becomes an OOD element, we can code it by adding the "Color"

        for lvl_type, n_lvls in enumerate(num_levels_each_type):  # 0: in_distr, 1: ood
            # Define the env and ids to be used
            if lvl_type == 0:  # In-Distribution
                level_ids_list = self.level_ids
                aux_env = self.env

            else:  # OOD
                level_ids_list = self.level_ids_ood
                aux_env = self.env_ood

            for e in range(n_lvls):
                
                # Initialize environment with next_seed
                next_seed = level_ids_list[e]
                aux_env.envs[0].seed(int(next_seed))
                obs = [aux_env.envs[0].reset()]
                steps = 0
                
                all_ood_states_one_eps = []
                num_oods_in_eps = 0
                num_ood_correctly_detected = 0
                while True:

                    steps += 1
                    
                    ### Process the state and get the action ###
                    preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                    with torch.no_grad():
                        if self.separated_actor_critic:
                            distribution = self.acmodel[0](preprocessed_obs)
                            value = self.acmodel[1](preprocessed_obs)
                        else:
                            if self.ood_module:
                                distribution, activations, value = self.acmodel(obs=preprocessed_obs,
                                                                                ood_method=self.ood_module.name)
                                
                                #all_activations_one_eps.append(self.ood_module.format_one_step_activations(activations))
                            else:
                                distribution, value = self.acmodel(obs=preprocessed_obs)

                    action = distribution.sample()
                    
                    ### Check if the state is an OOD ###
                    # To do so, check if the state has any of the OOD elements
                    current_state_is_ood_ground_truth = 0  # 1 is OOD, 0 is In-Distribution
                    if ood_detection_performance_eval:

                        # if window_visible:  # To save the window 
                        #     img = aux_env.render('rgb_array')

                        for id_ood in ood_ids_list:
                            if len(np.where(obs[0]['image'][:, :, 0] == id_ood)[0]) >= 1:
                                num_oods_in_eps += 1
                                current_state_is_ood_ground_truth = 1
                                break
                        if not current_state_is_ood_ground_truth:
                            for color_ood in ood_colors_list:
                                if len(np.where(obs[0]['image'][:, :, 1] == color_ood)[0]) >= 1:
                                    num_oods_in_eps += 1
                                    current_state_is_ood_ground_truth = 1
                                    break
                    
                    ### Take a step in the environment ###
                    #obs, reward, done, _ = aux_env.step(actions=action, seeds=[next_seed])
                    obs, reward, done, _ = aux_env.step(actions=action, seeds=len(seeds)*[None])
                    reward = np.array(reward)
                    done = np.array(done)

                    ### Use the OOD method to detect if the state (BEFORE) is OOD ###
                    if self.method_type == "ood":
                        if self.ood_module.which_internal_activations == 'observations':
                            current_obs = preprocessed_obs.image
                            next_obs = self.preprocess_obss(obs, device=self.device).image
                            activations = [current_obs, next_obs]
                        ood_state = self.ood_module.compute_ood_decision_on_one_step(activations, actions=action.cpu().numpy())
                        all_ood_states_one_eps.append(ood_state)
                        

                        # Check if the OOD state has been detected correctly
                        if ood_detection_performance_eval:
                            # Count an OOD state as correctly detected if it is detected (first condition) and it is OOD (second condition)
                            if (ood_state == current_state_is_ood_ground_truth) and (current_state_is_ood_ground_truth == 1):
                                num_ood_correctly_detected += 1

                    if done:
                        
                        ### Compute the OOD value of the episode depending on the method ###
                        if self.method_type == "ood":
                            
                            all_ood_states_one_eps = np.array(all_ood_states_one_eps)

                            ## Different options to compute the OOD value of an episode ##
                            if ood_detection_performance_eval:
                                option = 2
                            else:
                                option = 1
                            # Option 1: Mean of the OOD values (normalizes by the number of steps)
                            if option == 1:
                                ood_value_of_eps = np.mean(all_ood_states_one_eps)
                            
                            # Option 2: Absolute count of OOD states
                            elif option == 2:
                                ood_value_of_eps = np.sum(all_ood_states_one_eps)

                            else:
                                raise ValueError("The option to compute the OOD value of an episode is not valid.")

                        elif self.method_type == "regret":
                            raise NotImplementedError("Regret method is not implemented yet.")

                        elif self.method_type == "performance":
                            # If the reward is 0, it means the level has not been resolved so it is considered OOD
                            if reward < 0.0001:
                                ood_value_of_eps = steps  # The OOD value is the number of steps, as we only can evaluate the whole episode
                            else:
                                ood_value_of_eps = 0
                        else:
                            raise ValueError("The method type is not valid. Please choose between ood, regret or performance.")   

                        ### Add the string to identify the ground truth of the level ###
                        if lvl_type == 0:
                            in_distr_or_ood = 'ind'
                        else:
                            in_distr_or_ood = 'ood'

                        ### Add the OOD value to the dictionary ###
                        ood_value_of_eps = float(ood_value_of_eps)
                        if ood_detection_performance_eval:
                            self.ood_score_of_levels[str(next_seed) + f'_{in_distr_or_ood}'] = (num_oods_in_eps, num_ood_correctly_detected, ood_value_of_eps, steps)

                        else:
                            self.ood_score_of_levels[str(next_seed) + f'_{in_distr_or_ood}'] = ood_value_of_eps                 
                        
                        break
            
            ### Finished with the levels of one type (in_distr or ood) ###
            print(f'Finished with the {in_distr_or_ood} levels')

        if ood_detection_performance_eval:
            return  # Just return in the case of OOD detection performance evaluation

        if self.method_type == "ood":
            # Convert the values of the dictionary to probabilities of being sampled
            # First compute the sum of the values
            sum_values = 0
            for values in self.ood_score_of_levels.values():
                sum_values += values
            # Then divide each value by the sum
            for key, value in self.ood_score_of_levels.items():
                self.ood_score_of_levels[key] = value / sum_values

    def generate_in_distribution_and_thresholds(self, tpr):
        """
            Generate the distribution of the in-distribution levels and the thresholds.
            Save them to a file in the storage folder, in the directory
            of the root or pretrained model (specified in args.model)

        """
        seeds = self.current_eps_seeds
        if self.ood_module.per_class:
            all_activations = [[] for _ in range(self.env.action_space.n)]
        else:
            all_activations = []
        level_count = 0
        # Loop over the in-distribution levels to get the activations
        for lvl in self.level_ids:
            # Initialize environment with next_seed
            self.env.envs[0].seed(int(lvl))
            obs = [self.env.envs[0].reset()]
            steps = 0
            
            while True:
                steps += 1

                preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                with torch.no_grad():
                    if self.separated_actor_critic:
                        raise NotImplementedError("Separated actor-critic is not implemented yet.")
                        distribution = self.acmodel[0](preprocessed_obs)
                        value = self.acmodel[1](preprocessed_obs)
                    else:
                        # TODO: Only working with ACModelRIDE for the moment
                        distribution, activations, value = self.acmodel(obs=preprocessed_obs, 
                                                                        ood_method=self.ood_module.name)

                action = distribution.sample()

                obs, reward, done, _ = self.env.step(actions=action, seeds=len(seeds)*[None])

                # For the forward dynamics methods
                if self.ood_module.which_internal_activations == 'observations':
                    current_obs = preprocessed_obs.image
                    next_obs = self.preprocess_obss(obs, device=self.device).image  #obs[0]['image']
                    activations = [current_obs, next_obs]

                self.ood_module.append_one_step_activations_to_list(all_activations, activations, action)

                reward = np.array(reward)
                done = np.array(done)

                if done:
                    level_count += 1
                    if level_count % 100 == 0:
                        print(f"Level {level_count}/{len(self.level_ids)} done")
                    break
        
        # Generate the clusters if the method requires it
        if self.ood_module.distance_method:
            self.ood_module.generate_clusters(all_activations)

        # Generate the thresholds (Not used in the actual implementation, but it is useful for the future)
        self.ood_module.thresholds = self.ood_module.generate_thresholds(all_activations, tpr=tpr)

    def load_ood_method(self, ood_method) -> str:
        """
            Load the OOD method's class in the ood module and return the method type string
        """
        from constants import OOD_METHODS
        from utils.ood_utils import select_ood_method
        if ood_method in OOD_METHODS:
            self.ood_module = select_ood_method(ood_method)
        else:
            raise ValueError(f"The OOD method is not valid. Please choose between {OOD_METHODS}")
        return 'ood'

    def generate_metrics_for_detection_performance(self, ood_detection_range=9):
        """
            Create a pool of levels to be used during offline training
        """

        num_levels_each_type = [len(self.level_ids), len(self.level_ids_ood)]
        seeds = self.current_eps_seeds

        ### CODE FOR COUNTING OOD STATES ###
        ood_ids_list = []
        ood_colors_list = []
        ind_env_id = self.env.envs[0].spec.id.split('-')[1]
        ood_env_id = self.env_ood.envs[0].spec.id.split('-')[1]
        # We have to now in advance the OOD elements introduced in the environment
        if ("Lava" in ood_env_id) or ("L" in ood_env_id) or ("Quicksand" in ood_env_id):
            if not ("Lava" in ind_env_id):  # This is for the case when the in_distr env is Lava and the OOD env is Lava+Ball
                ood_ids_list.append(9)
        if ("Ball" in ood_env_id) or ("B" in ood_env_id):
            if not ("Ball" in ind_env_id):  # This is for the case when the in_distr env is Ball and the OOD env is Lava+Ball
                ood_ids_list.append(6)
        # In case the color becomes an OOD element, we can code it by adding the "Color"
                
        # Initialize the dictionary to store the activations of each episode
        ood_results_per_level = OrderedDict()

        # Per unique step
        ood_score_per_unique_step = OrderedDict()
        ground_truth_per_unique_step = OrderedDict()
        ood_results_per_unique_step = OrderedDict()

        # Count number of levels, success rate and mean return
        reward_and_success_metrics = []

        # For quicksand env
        retention_counter = 0
        has_left_the_quicksand = True
        
        # Loop over the levels
        for lvl_type, n_lvls in enumerate(num_levels_each_type):  # 0: in_distr, 1: ood
            # Define the env and ids to be used
            if lvl_type == 0:  # In-Distribution
                level_ids_list = self.level_ids
                aux_env = self.env

            else:  # OOD
                level_ids_list = self.level_ids_ood
                aux_env = self.env_ood

            num_levels = 0
            num_levels_successfully_solved = 0
            sum_of_return = 0.0

            for e in range(n_lvls):
                
                # Initialize environment with next_seed
                next_seed = level_ids_list[e]
                aux_env.envs[0].seed(int(next_seed))
                obs = [aux_env.envs[0].reset()]
                steps = 0
                
                scores_one_eps = []
                ground_truth_one_eps = []
                num_oods_in_eps = 0
                while True:

                    steps += 1
                    
                    ### Process the state and get the action ###
                    preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                    with torch.no_grad():
                        if self.separated_actor_critic:
                            distribution = self.acmodel[0](preprocessed_obs)
                            value = self.acmodel[1](preprocessed_obs)
                        else:
                            if self.ood_module:
                                distribution, activations, value = self.acmodel(obs=preprocessed_obs,
                                                                                ood_method=self.ood_module.name)
                                
                                #scores_one_eps.append(self.ood_module.format_one_step_activations(activations))
                            else:
                                distribution, value = self.acmodel(obs=preprocessed_obs)

                    action = distribution.sample()
                    
                    ### Check if the state is an OOD ###
                    # To do so, check if the state has any of the OOD elements
                    current_state_is_ood_ground_truth = 0  # 1 is OOD, 0 is In-Distribution
                    dist_to_ood = 0  # By default, the distance to the OOD is 0 (In-Distribution)
                    for id_ood in ood_ids_list:
                        ood_object_positions = np.argwhere(obs[0]['image'][:, :, 0] == id_ood)
                        if len(ood_object_positions) >= 1:
                            num_oods_in_eps += 1
                            current_state_is_ood_ground_truth = 1
                            # Compute the distance to the ood in manhattan distance. Agent's position is (3,6)
                            l1_distances = [abs(pos[0] - 3) + abs(pos[1] - 6) for pos in ood_object_positions]
                            # Find the minimum L1 distance
                            dist_to_ood = min(l1_distances) if l1_distances else None

                    ### Take a step in the environment ###
                    #obs, reward, done, _ = aux_env.step(actions=action, seeds=[next_seed])
                    obs, reward, done, _ = aux_env.step(actions=action, seeds=len(seeds)*[None])
                    reward = np.array(reward)
                    done = np.array(done)

                    ### Use the OOD method to detect if the state is OOD ###
                    if self.method_type == "performance":
                        pass  # Nothing to do here

                    elif self.method_type == "regret":
                        raise NotImplementedError("Regret method is not implemented yet.")
                    
                    else:  # OOD

                        current_obs = preprocessed_obs.image

                        if self.ood_module.which_internal_activations == 'observations':
                            # For the forward dynamics methods
                            next_obs = self.preprocess_obss(obs, device=self.device).image  #obs[0]['image']
                            activations = [current_obs, next_obs]

                        scores_one_eps.append(self.ood_module.compute_ood_score_on_one_step(activations, action))
                        ground_truth_one_eps.append(current_state_is_ood_ground_truth)

                        dict_key = str(action.cpu().numpy().item()) + str(current_obs.cpu().numpy().flatten().tolist())
                        if not dict_key in ood_score_per_unique_step.keys():
                            ood_results_per_unique_step[dict_key] = [ground_truth_one_eps[-1], dist_to_ood, scores_one_eps[-1]]
                        else:
                            # Check if the ground truth and the score are the same when the same state is seen
                            if ood_results_per_unique_step[dict_key][0] != ground_truth_one_eps[-1]:
                                raise ValueError("The ground truth of the same state is different in different episodes.")
                            if ood_results_per_unique_step[dict_key][1] != scores_one_eps[-1]:
                                raise ValueError("The score of the same state is different in different episodes.")

                    if done:
                        # Count number of levels, success rate and mean return
                        num_levels += 1
                        if reward > 0:
                            num_levels_successfully_solved += 1
                        sum_of_return += reward.item()

                        ### Accumulate the activations of the episode ###
                        in_distr_or_ood = 'ind' if lvl_type == 0 else 'ood'
                        info_one_eps = np.array([ground_truth_one_eps, scores_one_eps]).transpose()
                        ood_results_per_level[str(next_seed) + f'_{in_distr_or_ood}'] = info_one_eps
                        break
            
            ### Finished with the levels of one type (in_distr or ood) ###
            print(f'Finished with the {in_distr_or_ood} levels')
            reward_and_success_metrics.append([num_levels, num_levels_successfully_solved, num_levels_successfully_solved/num_levels,
                                               sum_of_return, sum_of_return/num_levels, sum_of_return/num_levels_successfully_solved])

        # Per step results to one array
        ood_results_per_unique_step = np.array(list(ood_results_per_unique_step.values()))

        return ood_results_per_level, ood_results_per_unique_step, reward_and_success_metrics


    @abstractmethod
    def update_parameters(self):
        pass
