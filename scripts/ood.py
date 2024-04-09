import argparse
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

import utils
from utils.ood_utils import select_ood_method
from torch_ac.utils.penv import ParallelEnv
from constants import PATH_OOD_DIR, OOD_METHODS, IM_MOD_METHODS
from gym_minigrid.minigrid import Quicksand

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--env", default='MiniGrid-NumpyMapMultiRoomPartialView-v0',
                    help="name of the environment to be run")
parser.add_argument("--env-list", nargs="+" , default=[],
                    help="subset of files that we are going to use")
parser.add_argument("--num_episodes", type=int, default=1,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--separated_networks", default=0, type=int,
                    help="set if we use two different NN for actor and critic")
parser.add_argument("--use-gpu", type=int, default=0)
parser.add_argument("--gpu-id", type=int, default=0)

parser.add_argument("-v", "--visualize", action='store_true', default=False)

# OOD related arguments
parser.add_argument("--ood_eval", action='store_true', default=False,
                    help='To run the model in OOD evaluation mode.' \
                         'Requires --ood_method defined and thresholds previously generated and stored')
parser.add_argument("--load_thresholds", action='store_true', default=False, \
                    help='To load the thresholds from a file')
parser.add_argument("--threshold_file", default='', type=str,
                    help="File with the thresholds to use for OOD evaluation," \
                         " must be a json storaged in ./storage_ood/")
parser.add_argument("-thr", "--threshold_generation", action='store_true', default=False,
                    help='To generate the threshold or thresholds for OOD generation')
parser.add_argument("--ood_method", type=str, choices=OOD_METHODS,
                    help='Specify the OOD method to use for evaluation')
parser.add_argument("--save_activations", action='store_true', default=False,
                    help='To save activations of the last layer of the network into a file')
parser.add_argument("--load_activations", action='store_true', default=False,
                    help='To load activations of the last layer of the network from a file' \
                         'It uses the model and the env to look for them')

# Explainability related arguments
parser.add_argument("--explainer", type=str, default='',
                    help='Define the method to attribute pixels in the visualization')

args = parser.parse_args()

# Checks
if args.env_list:
    assert args.env == 'MiniGrid-NumpyMapMultiRoomPartialView-v0', 'Env must be MiniGrid-NumpyMapMultiRoomPartialView-v0 if env_list is specified'
    #raise ValueError('You have to specify either an env or an env_list')

if args.threshold_generation:
    assert args.ood_eval == False, 'Cannot generate thresholds and evaluate at the same time'
    assert args.visualize == False, 'Cannot visualize and generate thresholds at the same time'
    #assert args.num_episodes >= 10, 'Need at least 10 episodes to generate thresholds'

if args.load_activations and args.save_activations:
    raise ValueError('Cannot save and load activations at the same time')

if args.ood_eval:
    assert args.load_activations == False, 'Cannot load activations and evaluate at the same time'
    assert args.save_activations == False, 'Cannot save activations and evaluate at the same time'
    assert args.visualize == True, 'For the moment we are going to visualize the OOD evaluation'
    assert args.load_thresholds == True, 'You have to load the thresholds to evaluate OOD'
    #assert args.threshold_file != '', 'You have to specify a threshold file in ./storage_ood/ to evaluate OOD'

if args.explainer:
    if not args.visualize:
        args.visualize = True
        print('Visualize set to True to be able to explain the agent')
    

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda:"+str(args.gpu_id) if args.use_gpu else "cpu")
print(f"Device: {device}\n")

# ******************************************************************************
# Generate ENVIRONMENT DICT
# ******************************************************************************
# MiniGrid-MultiRoom-N7-S4-v0
env_dict = {}
env_list = [name + '.npy' for name in args.env_list] #add file extension
env_dict[args.env] = env_list
print('Env Dictionary:',env_dict)

# Load environments
env = utils.make_env(env_dict, args.seed)
print("Environments loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space.n, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text,
                        separated_networks=args.separated_networks,
                        explainer=args.explainer)
print("Agent loaded\n")

# Load thresholds and initialize ood_decision
if args.ood_method:

    # Load OOD method
    ood_method = select_ood_method(args.ood_method)

    # Initialize ood_state
    ood_state = False

    if args.load_thresholds:
        ood_method.load_ood_method_info(PATH_OOD_DIR, model_name=args.model)
        print(f'Thresholds loaded from {PATH_OOD_DIR}')

        # Print clusters if distance method
        if ood_method.distance_method:
            print('Clusters:')
            print(ood_method.clusters)

    if args.ood_method in IM_MOD_METHODS:
        from torch_ac.utils.intrinsic_motivation import RIDEModule
        from torch_ac.utils.im_models import EmbeddingNetwork_RAPID, EmbeddingNetwork_RIDE, \
                                        InverseDynamicsNetwork_RAPID, InverseDynamicsNetwork_RIDE, \
                                        ForwardDynamicsNetwork_RAPID, ForwardDynamicsNetwork_RIDE
        num_actions = env.action_space.n
        im_module = RIDEModule(emb_network = EmbeddingNetwork_RIDE(),
                               inv_network = InverseDynamicsNetwork_RIDE(num_actions=num_actions, device=utils.device),
                               forw_network = ForwardDynamicsNetwork_RIDE(num_actions=num_actions, device=utils.device),
                               device = utils.device)
        # Load status with the weights of RIDE
        status = utils.get_status(model_dir)
        im_module.state_embedding.load_state_dict(status["im_mod_state_embedding_weights"])
        im_module.optimizer_state_embedding.load_state_dict(status["im_mod_state_embedding_optimizer"])
        im_module.inverse_dynamics.load_state_dict(status["im_mod_inverse_dynamics_weights"])
        im_module.optimizer_inverse_dynamics.load_state_dict(status["im_mod_inverse_dynamics_optimizer"])
        im_module.forward_dynamics.load_state_dict(status["im_mod_forward_dynamics_weights"])
        im_module.optimizer_forward_dynamics.load_state_dict(status["im_mod_forward_dynamics_optimizer"])

        if args.ood_method in IM_MOD_METHODS:
            ood_method.im_module = deepcopy(im_module)  # TODO: COMPROBAR SI ESTA EN LA GRAFICA O EN LA CPU

else:
    ood_state = None

if args.visualize:
    env.render('human')  # Create a window to view the environment

# In case we load activations, we do not need to run the agent
if args.load_activations:
    print('Loading activations...')
    all_activations = ood_method.load_activations(save_path=PATH_OOD_DIR, model_name=args.model)
    print('Done loading activations')

# Run the agent normally, either for evaluation or for threshold generation
else:
    # Run agent
    start_time = time.time()

    # Initialize

    if ood_method.per_class:
        all_activations = [[] for _ in range(env.action_space.n)]
    else:
        all_activations = []

    has_left_the_quicksand = True
    retention_counter = 0
    in_lava = False
    if hasattr(env, 'quicksand_prob'):
        in_quicksand_env = True
    else:
        in_quicksand_env = False
    
    # for N episodes
    for e in range(args.num_episodes):

        # Episodic things
        obss = env.reset()
        env.seed(args.seed + int(e))
        steps = 0
        episode_info = {}
        counter_of_consecutive_ood_states = 0

        # TODO: Now im only rendering the first time and then the render
        #   is done by the explainability part
        first_render = 0
        
        # one episode
        while True:
            # For explainer, only render the first step of the episode
            #  to load the plot
            if args.explainer:
                if first_render == 0:
                    if args.visualize:
                        env.render('human')
                    first_render += 1
            else:
                if args.visualize:
                        if args.ood_eval:
                            env.render('human', ood_state=ood_state)
                        else:
                            env.render('human')
            steps += 1

            if args.explainer:

                actions, activations, vext, attribution = agent.get_actions(obss, ood_method=args.ood_method)

                env.render('human', attribution=attribution, action_taken=actions[0])

            ### OOD evaluation or threshold generation, extract internal activations ###
            elif args.ood_method:

                actions, activations, vext = agent.get_actions(obss, ood_method=args.ood_method)

                action_taken = actions[0]

                ## Threshold generation ##
                if args.threshold_generation:

                    # For the forward dynamics methods
                    if ood_method.which_internal_activations == 'observations':
                                current_obs = deepcopy(obss['image'])
                    else:
                        ood_method.append_one_step_activations_to_list(
                            all_activations=all_activations,
                            one_step_activations=activations,
                            actions=actions
                        )
                ## OOD evaluation ##
                elif args.ood_eval:
                    
                    if ood_method.which_internal_activations == 'observations':
                        current_obs = deepcopy(obss['image'])
                        
                    else:
                        ood_state = ood_method.compute_ood_decision_on_one_step(activations, actions)

                        print(f'Step {steps:03d}, Action taken: {action_taken}')
                        
                        # We have to render after the decision and before taking the next step
                        # to visualize correctly the real OOD state and then reset the
                        # ood_state variable to False for the next step
                        if ood_state:
                            env.render('human', ood_state=ood_state)
                            ood_state = False
                            counter_of_consecutive_ood_states += 1
                            if counter_of_consecutive_ood_states >= 5:
                                print(f'Episode {e} of {args.num_episodes} finished in ',steps)
                                break
                        else:
                            counter_of_consecutive_ood_states = 0
            
            ### Normal case, just compute the actions and value ###
            else:
                actions, vext = agent.get_actions(obss)

            if has_left_the_quicksand:  # If the agent has left the ball, it can be blocked again
                if isinstance(env.grid.get(*env.agent_pos), Quicksand):  # Working only with balls for the moment
                    retention_counter += 1
                    if retention_counter < 3:
                        actions = torch.tensor([6], device=device, dtype=torch.int)
                    else:  # Time to unblock the agent in the ball
                        retention_counter = 0
                        has_left_the_quicksand = False
                        #action = torch.tensor([2], device=self.device, dtype=torch.int)  # Move forward
            else:
                if not isinstance(env.grid.get(*env.agent_pos), Quicksand):
                    has_left_the_quicksand = True

            ### Take the next step in the environment ###
            obss, rewards, dones, _ = env.step(actions)
            episode_info[steps] = [rewards,vext]

            ### For the forward dynamics methods we need to previously compute the next observation ###
            if ood_method.which_internal_activations == 'observations':
                
                next_obs = deepcopy(obss['image'])

                if args.threshold_generation:
                    ood_method.append_one_step_activations_to_list(
                        all_activations=all_activations,
                        one_step_activations=[current_obs, next_obs],
                        actions=actions
                    )
                elif args.ood_eval:
                    ood_state = ood_method.compute_ood_decision_on_one_step([current_obs, next_obs], actions)
                    print(f'Step {steps:03d}, Action taken: {action_taken}')
                    
                    # In the case of the OOD evaluation of the forward dynamics methods, we render the next state
                    if ood_state:
                        env.render('human', ood_state=ood_state)
                        ood_state = False
                        counter_of_consecutive_ood_states += 1
                        if counter_of_consecutive_ood_states >= 5:
                            print(f'Episode {e} of {args.num_episodes} finished in ',steps)
                            break
                    else:
                        counter_of_consecutive_ood_states = 0
                else:
                    raise ValueError('This should not happen')

            if dones:
                print(f'Episode {e} of {args.num_episodes} finished in ',steps)
                break

    end_time = time.time()

if args.save_activations:
    print('Saving activations...')
    ood_method.save_activations(all_activations=all_activations, save_path=PATH_OOD_DIR, model_name=args.model)
    print('Done saving activations')

if args.threshold_generation:

    print('Generating thresholds for OOD detection...')
    #all_activations = np.array(all_activations)

    if ood_method.distance_method:
        ood_method.generate_clusters(all_activations)

    ood_method.thresholds = ood_method.generate_thresholds(all_activations, tpr=0.95)
    ood_method.save_ood_method_info(save_path=PATH_OOD_DIR, model_name=args.model)

    print('Done generating thresholds')
