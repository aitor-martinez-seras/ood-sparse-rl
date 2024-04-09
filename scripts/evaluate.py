import argparse
import time
from pathlib import Path
from csv import writer

import torch
from torch_ac.utils.penv import ParallelEnv
from gym_minigrid.minigrid import Quicksand
import numpy as np

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--env", default='MiniGrid-NumpyMapMultiRoomPartialView-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--env-list", nargs="+" , default=[],
                    help="subset of files that we are going to use")
parser.add_argument("--episodes", type=int, default=100,
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
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("-v", "--visualize", action='store_true', default=False)
parser.add_argument("--use-gpu", type=int, default=0)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--save", action='store_true', default=False)
parser.add_argument("--random_agent", action='store_true', default=False)

args = parser.parse_args()

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
if not args.random_agent:
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space.n, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text,
                        separated_networks=args.separated_networks)
    print("Agent loaded\n")

if args.visualize:
    env.render('human') # Create a window to view the environment

# Run agent
start_time = time.time()

# Metrics
number_of_levels_played = 0
number_of_levels_solved = 0
total_return = 0

# Check if we have a csv file with the evaluation already
if args.random_agent:
    csv_path = Path(f"ood_storage/performance_evaluation_random_agent.csv")
else:
    model_name = Path(model_dir).name[:-2]
    csv_path = Path(f"ood_storage/performance_evaluation_{model_name}.csv")

# for N episodes
for e in range(args.num_episodes):

    # Episodic things
    obss = env.reset()
    env.seed(args.seed + int(e))
    steps = 0
    episode_info = {}
    number_of_levels_played += 1

    # one episode
    while True:
        if args.visualize:
            env.render('human')
        if args.save:
            img = env.render('rgb_array')
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(f'ood_storage/{args.env_list[0]}.pdf', bbox_inches='tight')
            plt.close()
            exit()

        steps += 1

        if args.random_agent:
            actions = np.random.randint(0, 7, size=[1])  # Random actions
            vext = 0
        else:
            actions, vext = agent.get_actions(obss)
        

        obss, rewards, dones, _ = env.step(actions)
        episode_info[steps] = [rewards,vext]

        if dones:
            print('Episode finished in ',steps)
            total_return += rewards
            if rewards > 0.01:
                number_of_levels_solved += 1
            break

print(f'Number of levels played: {number_of_levels_played}')
print(f'Number of levels solved: {number_of_levels_solved}')
print(f'Success rate: {number_of_levels_solved/number_of_levels_played}')
print(f'Mean return: {total_return/number_of_levels_played}')


# # OPTION 1: Using pandas
# if csv_path.exists():
#     print('CSV file already exists, loading it')
#     df = pd.read_csv(csv_path)
#     print(df)
# else:
#     print('CSV file does not exist, creating it')
#     df = pd.DataFrame(columns=COLUMNS)
# # Add info to dataframe
# df = df.append({'env': list(env_dict.keys())[0],
#                 'seed': args.seed,
#                 'num_episodes_played': number_of_levels_played,
#                 'num_episodes_solved': number_of_levels_solved,
#                 'success_rate': number_of_levels_solved/number_of_levels_played,
#                 'mean_return': total_return/number_of_levels_played},
#                 ignore_index=True)
# # Save dataframe
# df.to_csv(csv_path, index=False)

# OPTION 2: Using csv writer
# List that we want to add as a new row
if args.random_agent:
    return_in_solved_eps = total_return/number_of_levels_solved if number_of_levels_solved > 0 else 0
    data = [list(env_dict.keys())[0], args.seed, number_of_levels_played, number_of_levels_solved, number_of_levels_solved/number_of_levels_played,
        total_return, total_return/number_of_levels_played, return_in_solved_eps]
else:
    data = [list(env_dict.keys())[0], args.seed, number_of_levels_played, number_of_levels_solved, number_of_levels_solved/number_of_levels_played,
            total_return, total_return/number_of_levels_played, total_return/number_of_levels_solved]
COLUMNS = ['env', 'seed', 'num_episodes_played', 'num_episodes_solved', 'success_rate', 'sum_of_returns', 'mean_return', 'mean_return_in_solved_eps']

# Check if csv exists
if not csv_path.exists():
    # Create a file object for this file
    with open(csv_path, 'w') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
 
        # Create the headers row and pass the list as an argument into the writerow()
        writer_object.writerow(COLUMNS)
        writer_object.writerow(data)
 
        #Close the file object
        f_object.close()
else:
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(csv_path, 'a') as f_object:
        # Pass this file object to csv.writer() and get a writer object
        writer_object = writer(f_object)
        # Pass the list as an argument into the writerow()
        writer_object.writerow(data)
        # Close the file object
        f_object.close()
        
end_time = time.time()
