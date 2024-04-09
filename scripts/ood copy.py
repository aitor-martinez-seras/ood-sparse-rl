import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise as sk_pairwise

import utils
from torch_ac.utils.penv import ParallelEnv
from constants import PATH_OOD_DIR, PATH_PLOTS, OOD_METHODS, LOGITS_METHODS, CONV_METHODS, DISTANCE_METHODS

# def write_pickle(an_object, path_to_file: Path):
#     print(f"Started writing object {type(an_object)} data into a .pkl file")
#     # store list in binary file so 'wb' mode
#     with open(path_to_file, 'wb') as fp:
#         pickle.dump(an_object, fp)
#         print('Done writing list into a binary file')


# def read_pickle(path_to_file: Path):
#     # for reading also binary mode is important
#     with open(path_to_file, 'rb') as fp:
#         an_object = pickle.load(fp)
#         return an_object

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

parser.add_argument("-v", "--visualize", action='store_true', default=False)

# OOD related arguments
parser.add_argument("--ood_eval", action='store_true', default=False,
                    help='To run the model in OOD evaluation mode.' \
                         'Requires --ood_method defined and thresholds previously generated and stored')
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
    assert args.num_episodes >= 10, 'Need at least 10 episodes to generate thresholds'

if args.load_activations and args.save_activations:
    raise ValueError('Cannot save and load activations at the same time')

if args.ood_eval:
    assert args.load_activations == False, 'Cannot load activations and evaluate at the same time'
    assert args.save_activations == False, 'Cannot save activations and evaluate at the same time'
    assert args.visualize == True, 'For the moment we are going to visualize the OOD evaluation'
    assert args.threshold_file != '', 'You have to specify a threshold file in ./storage_ood/ to evaluate OOD'

if args.explainer:
    if not args.visualize:
        args.visualize = True
        print('Visualize set to True to be able to explain the agent')
    

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {utils.device}\n")

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
if args.ood_eval:
    
    # Initialize ood_state
    ood_state = False

    # Load needed assests for the OOD evaluation
    if args.ood_method in LOGITS_METHODS:
        threshold_file_path = PATH_OOD_DIR / args.threshold_file
        thresholds = utils.read_json(threshold_file_path)
        print(f'Thresholds loaded from {threshold_file_path}')

    elif args.ood_method in CONV_METHODS:
        threshold_file_path = PATH_OOD_DIR / args.threshold_file
        threshold_and_centroids = utils.read_json(threshold_file_path)
        print(f'Thresholds and centroids loaded from {threshold_file_path}')
        threshold = threshold_and_centroids['threshold']
        centroids = np.array(threshold_and_centroids['centroid'])

else:
    ood_state = None

if args.visualize:
    env.render('human') # Create a window to view the environment

# In case we load activations, we do not need to run the agent
if args.load_activations:
    print('Loading activations...')
    if args.ood_method in LOGITS_METHODS:    
        all_logits_per_action = utils.read_json(PATH_OOD_DIR / f'logits_{args.model}_{args.env}.json')
    elif args.ood_method in CONV_METHODS:
        all_ftmaps = utils.read_npy(PATH_OOD_DIR / f'ftmaps_{args.model}_{args.env}.npy')
    print('Done loading activations')

    #if args.ood_method in DISTANCE_METHODS:


# Run the agent normally, either for evaluation or for threshold generation
else:
    # Run agent
    start_time = time.time()

    # Initialize 
    # MSP
    all_logits_per_action = [[] for _ in range(env.action_space.n)]
    # TSNE
    #all_ftmaps_per_action = [[] for _ in range(env.action_space.n)]
    all_ftmaps = []
    all_ftmaps_posible_oods = []
    
    # for N episodes
    for e in range(args.num_episodes):

        # Episodic things
        obss = env.reset()
        env.seed(args.seed + int(e))
        steps = 0
        episode_info = {}

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

            # TODO: Implementar los OOD methods con clases para simplificar el codigo
            # OOD case (either evaluation or threshold generation)
            elif args.ood_method:

                actions, activations, vext = agent.get_actions(obss, ood_method=args.ood_method)

                action_taken = actions[0]

                if args.threshold_generation:

                    if args.ood_method in LOGITS_METHODS:
                        logits = activations
                        # OPT-1: Taking the max of the logits for each action
                        all_logits_per_action[action_taken].append(logits[0, action_taken].item())
                        # OPT-2: 
                        # all_logits_per_action[actions[0]].append(logits[0, actions[0]])
                    
                    elif args.ood_method in CONV_METHODS:
                        conv_output = activations
                        # Take the 0 position as the batch size is 1
                        all_ftmaps.append(conv_output[0].cpu().numpy())
                    
                elif args.ood_eval:

                    if args.ood_method in LOGITS_METHODS:

                        if thresholds[action_taken] == 0:
                            print(f'WARNING: Threshold for action {action_taken} is 0. Marking it as OOD, as it has never been trained on')
                            ood_state = True

                        elif args.ood_method == 'msp':
                            # We are going to use the logit and threshold for the action taken
                            logits = activations
                            ood_state = logits[0, action_taken] < thresholds[action_taken]
                        
                        elif args.ood_method == 'energy':
                            raise NotImplementedError
                    
                    elif args.ood_method in CONV_METHODS:

                        conv_output = activations

                        if args.ood_method == 'l1':
                            
                            distances = sk_pairwise.pairwise_distances(
                                metric='manhattan',
                                X=conv_output.cpu().numpy(),
                                Y=centroids
                            )
                            # In this case the mean is redundant as there is 
                            # only one cluster or centroid (the mean)
                            distance = distances.min()
                            print(f'Threshold: {threshold}')
                            print(f'Distance: {distance}')

                            if distance > threshold:
                                ood_state = True

                        if args.ood_method == 'l2':
                            
                            distances = sk_pairwise.pairwise_distances(
                                metric='euclidean',
                                X=conv_output.cpu().numpy(),
                                Y=centroids
                            )
                            # In this case the mean is redundant as there is 
                            # only one cluster or centroid (the mean)
                            distance = distances.min()
                            print(f'Threshold: {threshold}')
                            print(f'Distance: {distance}')

                            if distance > threshold:
                                ood_state = True
                    
                    else:
                        raise ValueError(f'Unknown OOD method {args.ood_method}')
                    
                    # We have to render after the decision and before taking the next step
                    # to visualize correctly the real OOD state and then reset the
                    # ood_state variable to False for the next step
                    if ood_state:
                        env.render('human', ood_state=ood_state)
                        ood_state = False
            
            # Normal case
            else:
                actions, vext = agent.get_actions(obss)

            obss, rewards, dones, _ = env.step(actions)
            episode_info[steps] = [rewards,vext]

            if dones:
                print(f'Episode {e} of {args.num_episodes} finished in ',steps)
                break

    end_time = time.time()

if args.save_activations:
    print('Saving activations...')
    if args.ood_method in LOGITS_METHODS:
        utils.write_json(all_logits_per_action, PATH_OOD_DIR / f'logits_{args.model}_{args.env}.json')
    elif args.ood_method in CONV_METHODS:
        utils.write_npy(all_ftmaps, PATH_OOD_DIR / f'ftmaps_{args.model}_{args.env}.json')
    print('Done saving activations')

if args.threshold_generation:

    print('Generating thresholds for OOD detection...')
    if args.ood_method == 'msp':
        
        for idx_action in range(len(all_logits_per_action)):
            if len(all_logits_per_action[idx_action]) < 100:
                print(f'WARNING: Only {len(all_logits_per_action[idx_action])} logits obtained for action {idx_action}')
                all_logits_per_action[idx_action] = np.array([])
            else:
                all_logits_per_action[idx_action] = np.array(all_logits_per_action[idx_action])

        print('Calculating thresholds')
        import matplotlib.pyplot as plt
        number_of_actions_with_info = len([x for x in all_logits_per_action if len(x) > 0])
        fig, axes = plt.subplots(1, number_of_actions_with_info, figsize=(15,5), sharey=True)
        for j, ax in enumerate(axes):
            ax.hist(all_logits_per_action[j], bins=50, density=True, label=f'Action {j}')
            #ax.hist(all_logits_per_action[j], bins=50, density=True, label=f'Action {j}')
            #ax.(torch.sort(all_logits_per_action[j], dim=-1, descending=False)[0], label=f'Action {j}')
            ax.set_title(f'Action {j}')
            ax.set_xlabel('Sorted logits')
            if j == 0:
                ax.set_ylabel('Density values')
            ax.set_ylim([0, 1])
        fig.suptitle('Logits per action')
        plt.savefig(PATH_PLOTS / 'logits_per_action.png')
        plt.show()
        
        # Nos quedamos con el percentil 5% de los episodios por clase
        # y asi obtenemos el threshold
        threshold_per_action = []
        for idx_action in range(len(all_logits_per_action)):
            if len(all_logits_per_action[idx_action]) > 100:
                threshold_per_action.append(np.quantile(all_logits_per_action[idx_action], 0.05))
            else:
                threshold_per_action.append(0)
        print('Thresholds calculated!')
        print('Saving thresholds...')
        utils.write_json(threshold_per_action, PATH_OOD_DIR / f'thresholds_{args.ood_method}_{args.model}_{args.env}.json')
        print('Done saving thresholds')

    elif args.ood_method in DISTANCE_METHODS:

        # For distance methods we have to save the centroids of the clustered feature maps
        #   or just the mean of all feature maps (one cluster) and the threshold or thresholds,
        #   depending on if we differentiate between actions (classes) or not

        if isinstance(all_ftmaps, list):
            all_ftmaps = np.array(all_ftmaps)
        
        if args.ood_method in CONV_METHODS:

            # Calculate the mean of the feature maps
            mean_ftmaps = np.mean(all_ftmaps, axis=0)

            if args.ood_method == 'l1':
                # Calculate the L1 distance between each feature map and the mean
                distances = distances = sk_pairwise.pairwise_distances(
                    metric='manhattan',
                    X=all_ftmaps,
                    Y=mean_ftmaps.reshape(1, -1)
                )

            elif args.ood_method == 'l2':
                distances = sk_pairwise.pairwise_distances(
                    metric='euclidean',
                    X=all_ftmaps,
                    Y=mean_ftmaps.reshape(1, -1)
                )
            else:
                raise ValueError(f'Unknown OOD method {args.ood_method}')

            # Nos quedamos con el percentil 5% de los episodios por clase
            # y asi obtenemos el threshold
            threshold = np.quantile(distances, 0.95)
            json_data = {
                'threshold': threshold,
                'centroid': [mean_ftmaps.tolist()]
            }
            utils.write_json(json_data, PATH_OOD_DIR / f'threshold_and_centroids_{args.ood_method}_{args.model}_{args.env}.json')

    elif args.ood_method == 'pca':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        all_ftmaps = np.array(all_ftmaps)
        pca_transform = PCA(n_components=2).fit(all_ftmaps)
        X_embedded_ind = pca_transform.transform(all_ftmaps)
        limit1 = len(all_ftmaps)
        X_embedded_ood = pca_transform.transform(all_ftmaps_posible_oods)
        plt.scatter(x=X_embedded_ind[:limit1, 0], y=X_embedded_ind[:limit1, 1], c='blue')
        plt.scatter(x=X_embedded_ood[limit1:, 0], y=X_embedded_ood[limit1:, 1], c='red')
        plt.show()
        print()
