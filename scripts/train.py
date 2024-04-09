import argparse
import time
import datetime
from copy import deepcopy
import sys
from pathlib import Path

import tensorboardX
import utils
import torch
# for Actor-Critic
from model import ActorModel_RAPID, CriticModel_RAPID, ACModelRIDE, InverseDynamicsNetwork
from torch_ac.algos import PPOAlgo

from constants import OOD_METHODS, IM_MOD_METHODS, LOGITS_METHODS
from utils.ood_utils import save_ood_detection_results, save_performance_metrics

# ******************************************************************************
# Parse arguments
# ******************************************************************************

parser = argparse.ArgumentParser()

## General parameters

# Logs related
parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=100, help="number of updates between two saves (default: 100, 0 means no saving)")
parser.add_argument("--visualize_levels", type=int, default=0, help="boolean-int variable that set whether we want to visualize the levels or not")
parser.add_argument("--save_im_module", type=int, default=0, help="boolean-int variable that set whether we want to save the intrinsic motivation module or not")
parser.add_argument("--load_im_module", type=int, default=0, help="boolean-int variable that set whether we want to load the intrinsic motivation module (if available) or not")

# Environment dependant
parser.add_argument("--env", default='MiniGrid-MultiRoom-N7-S4-v0',help="name of the environment to train on (REQUIRED)")
parser.add_argument("--env-list", nargs="+" , default=[],help="subset of files that we are going to use")
parser.add_argument("--seed", type=int, default=1,help="random seed (default: 1)")
parser.add_argument("--init_train_seed", type=int, default=0, help="set the initial levels/seed to TRAIN from. 0 sets to be random/infinite.")
parser.add_argument("--num_train_seeds", type=int, default=-1,help="number of seeds/levels used for training. -1 sets to be random/infinite.")
parser.add_argument("--init_test_seed", type=int, default=0, help="set the initial levels/seed to TEST from. 0 sets to be random/infinite.")
parser.add_argument("--num_test_seeds", type=int, default=-1,help="number of seeds used for evaluation")
parser.add_argument("--evaluation_interval", type=int, default=250000,help="number of frames between two evaluations")
# OOD related arguments
parser.add_argument("--env_ood", default='', help="name of the environment of ood to train and evaluate on")
parser.add_argument("--ood_sampling_ratio", type=float, default=0.0, help="the ratio of ood samples vs normal samples in the training set being used")
parser.add_argument("--ood_method", type=str, default='', help="method to use for OOD detection")
parser.add_argument("--ood_info_file", type=str, default="", help="file where thresholds and/or centroids are stored")
parser.add_argument("--new_dir_name", type=str, default="", help="new name for the directory where the model will be saved")
parser.add_argument("--init_train_seed_ood", type=int, default=0, help="set the initial levels/seed to TRAIN from. 0 sets to be random/infinite.")
parser.add_argument("--num_train_seeds_ood", type=int, default=-1,help="number of seeds/levels used for training. -1 sets to be random/infinite.")
parser.add_argument("--init_test_seed_ood", type=int, default=0, help="set the initial levels/seed to TEST from. 0 sets to be random/infinite.")
parser.add_argument("--num_test_seeds_ood", type=int, default=-1,help="number of seeds used for evaluation")
parser.add_argument("--intervene_after", type=int, default=-1,help="number of levels to be trained on before intervening with manual sampling")
parser.add_argument("--intervene_every", type=int, default=-1,help="number of levels to be sampled normally between each manual sampling")
parser.add_argument("--adaptation_strategy", type=str, default="", help="strategy to use for adaptation")
parser.add_argument("--save_in_distribution_info", type=int, default=0, help="boolean-int variable that set whether we want to save the in distribution info or not")
parser.add_argument("--load_in_distribution_info", type=int, default=0, help="boolean-int variable that set whether we want to load the in distribution info or not")
parser.add_argument("--evaluate_ood_detection_performance", type=int, default=0, help="boolean-int variable that set whether we want to evaluate the OOD detection performance or not")
parser.add_argument("--ood_detection_range", type=int, default=9, help="range of levels to evaluate the OOD detection performance")
parser.add_argument("--tpr", type=int, default=0.98, help="Useless for now")


## Parameters for intrinsic motivation
parser.add_argument("--intrinsic-motivation", type=float, default=0,help="specify if we use intrinsic motivation (int_coef) to face sparse problems")
parser.add_argument("--im-type", default='counts',help="specify if we use intrinsic motivation, which module/approach to use")
parser.add_argument("--int-coef-type", default='static',help="specify how to decay the intrinsic coefficient")
parser.add_argument("--normalize-intrinsic-bonus", type=int, default=0,help="boolean-int variable that set whether we want to normalize the intrinsic rewards or not")
parser.add_argument("--use-episodic-counts", type=int, default=0,help="divide intrinsic rewards with the episodic counts for that given state")
parser.add_argument("--use-only-not-visited", type=int, default=0,help="apply mask to reward only those states that have not been explored in the episode")

# Self-supervised loss with IDM
parser.add_argument("--use_ssl", type=int, default=0, help="used to apply self supervised loss through IDM")
parser.add_argument("--use_only_ssl", type=int, default=0, help="only ssl updates (not RL used)")
parser.add_argument("--ssl_update_freq", type=int, default=1, help="freq in which the updates with self-supervision are executed")
parser.add_argument("--load_ssl", type=int, default=0, help="laod weights")

## Select algorithm and generic configuration params
parser.add_argument("--algo", default="ppo",help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--procs", type=int, default=16,help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,help="number of frames of training (default: 1e7)")
parser.add_argument("--separated-networks", type=int, default=0,help="set if we use two different NN for actor and critic")


## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,help="batch size for PPO (default: 256)")
parser.add_argument("--nsteps", type=int, default=None,help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,help="learning rate (default: 0.0001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,help="add a GRU to the model to handle text input")


# GPU/CPU Configuration
parser.add_argument("--use-gpu", type=int, default=0,help="Specify to use GPU as device to bootstrap the training")
parser.add_argument("--gpu-id", type=int, default=-1,help="add a GRU to the model to handle text input")


args = parser.parse_args()

# ******************************************************************************
# AssertionError to ensure inconsistency problems
# ******************************************************************************
# LOGIC --> if it is true, no error thrown
assert ((args.env =='MiniGrid-NumpyMapFourRoomsPartialView-v0') and (len(args.env_list)>0)) \
        or args.env != 'MiniGrid-NumpyMapFourRoomsPartialView-v0', \
        'You have selected to use Pre-defined environments for training but no subfile specified'

assert (args.use_gpu==False) or (args.use_gpu and args.gpu_id != -1), \
        'Specify the device id to use GPU'

# OOD related
if args.env_ood != '':
    assert args.procs == 1, 'OOD environments not supported with multiple processes'
    assert args.num_test_seeds_ood > 0, 'We have to define a number of test seeds for OOD evaluation'

assert args.intervene_every != 0, '0 is not valid for intervene_every, as it would mean that we never sample from the pool of detected levels'

if args.adaptation_strategy == 'offline':
    assert args.env_ood != '', 'We have to use OOD environments if we define an adaptation strategy'
    assert args.num_train_seeds > 0, 'We have to define an speficic number of train seeds for the offline adaptation strategy'
    assert args.num_train_seeds_ood > 0, 'We have to define an speficic number of train seeds of OOD levels for the offline adaptation strategy'
    assert args.intervene_after == 0, 'We have to start intervening after 0 levels'
    assert args.intervene_every == 1, 'We have to intervene every level, as every level must be sampled from the pool of detected levels'

if args.adaptation_strategy == 'online':
    assert args.intervene_after > 0, 'We have to start intervening after some levels'
    assert args.intervene_every > 1, 'We have to intervene every some levels, but it can not be every level'

if args.adaptation_strategy == '':
    assert args.intervene_after == -1 and args.intervene_every == -1, 'We have to set intervene_after and intervene_every to -1 (default) if we are not using adaptation'

if args.intervene_after > 0 or args.intervene_every > 0:
    assert args.env_ood != '', 'We have to use OOD environments to intervene'

if args.save_in_distribution_info:
    assert args.load_in_distribution_info == False, 'We can not save and load in distribution info at the same time'

if args.evaluate_ood_detection_performance:
    assert args.adaptation_strategy == '', 'We only want to evaluate the OOD detection performance'

# ******************************************************************************

args.mem = args.recurrence > 1

# ******************************************************************************
# Set run dir
# ******************************************************************************

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)
print(model_dir)
# Change model dir name if desired. This is useful to load the status of a pretrained model that is in the model_name dir
pretrained_model_dir = ''
if args.new_dir_name != "":
    new_model_dir = utils.get_model_dir(args.new_dir_name)
    print(f"New dir for the model specificed, model status has been loaded from {model_dir}," \
          f" but now the model will be saved in {new_model_dir}")
    pretrained_model_dir = model_dir
    model_dir = new_model_dir


# ******************************************************************************
# Load loggers and Tensorboard writer
# ******************************************************************************
txt_logger = utils.get_txt_logger(model_dir)
if not args.evaluate_ood_detection_performance:  # Deactivate loggers if we are evaluating the OOD detection performance
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    if args.env_ood != '':
        csv_ood_file, csv_ood_logger = utils.get_csv_logger(model_dir, name_of_csv_file='log_ood.csv')

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# ******************************************************************************
# Set seed for all randomness sources
# ******************************************************************************
utils.seed(args.seed)

# ******************************************************************************
# Set device
# ******************************************************************************
device = torch.device("cuda:"+str(args.gpu_id) if args.use_gpu else "cpu")
txt_logger.info(f"Device: {device}\n")


# ******************************************************************************
# Generate ENVIRONMENT DICT
# ******************************************************************************
# MiniGrid-MultiRoom-N7-S4-v0
env_dict = {}
env_list = [name + '.npy' for name in args.env_list] #add file extension
env_dict[args.env] = env_list
print('Env Dictionary:', env_dict)
# OOD envs
env_dict_ood = {}
if args.env_ood != '':
    env_dict_ood[args.env_ood] = []
    print('Env Dictionary OOD:', env_dict_ood)

# ******************************************************************************
# Load environments
# ******************************************************************************
envs = []
for i in range(args.procs):
    envs.append(utils.make_env(env_dict, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")
# OOD envs
envs_ood = []
if len(env_dict_ood) > 0:
    for i in range(args.procs):
        envs_ood.append(utils.make_env(env_dict_ood, args.seed + 10000 * i))

# Define action_space
ACTION_SPACE = envs[0].action_space.n
txt_logger.info(f"ACTION_SPACE: {ACTION_SPACE}\n")

# ******************************************************************************
# Load training status
# ******************************************************************************
try:
    if pretrained_model_dir:
        status = utils.get_status(pretrained_model_dir)
    else:
        status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# ******************************************************************************
# Load observations preprocessor
# ******************************************************************************
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")


# ******************************************************************************
# Load model
# ******************************************************************************
separated_networks = args.separated_networks
# Use 1 AC network or separated Actor and Critic
if separated_networks:
    actor = ActorModel_RAPID(obs_space, ACTION_SPACE, args.mem)
    critic = CriticModel_RAPID(obs_space, ACTION_SPACE, args.mem)
    actor.to(device)
    critic.to(device)
    if "model_state" in status:
        actor.load_state_dict(status["model_state"][0])
        critic.load_state_dict(status["model_state"][1])
    txt_logger.info("Models loaded\n")
    txt_logger.info("Actor: {}\n".format(actor))
    txt_logger.info("Critic: {}\n".format(critic))
    # save as tuple
    acmodel = (actor,critic)
    # calculate num of model params
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    total_params = actor_params + critic_params
    print('***PARAMS:\nActor {}\nCritic {}\nTotal {}'.format(actor_params,critic_params,total_params))
else:
    use_intcoefs = 1 if args.int_coef_type == 'ngu'else 0
    acmodel = ACModelRIDE(obs_space, ACTION_SPACE, use_intcoefs,args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    total_params = sum(p.numel() for p in acmodel.parameters())
    print('***PARAMS UNIQUE AC (RIDE):',total_params)
# ******************************************************************************
# Set Intrinsic Motivation
# ******************************************************************************
txt_logger.info("Intrinsic motivation:{}\n".format(args.im_type))
# ******************************************************************************
# Load algorithm
# ******************************************************************************
self_supervised_model = None
args.use_ssl = 1 if args.load_ssl > 0 else args.use_ssl
args.use_ssl = 1 if args.use_only_ssl > 0 else args.use_ssl

if args.use_ssl or args.load_ssl:
    # 32 comes from the embedding size after the CNN of the AC model
    self_supervised_model = InverseDynamicsNetwork(input_size=32,
                                                   num_actions=ACTION_SPACE)
    self_supervised_model.to(device)

# if args.algo == "a2c":
#     algo = torch_ac.A2CAlgo(envs, acmodel, device, args.nsteps, args.discount, args.lr, args.gae_lambda,
#                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
#                             args.optim_alpha, args.optim_eps, preprocess_obss)
algo = PPOAlgo(envs, acmodel, device, 
                    args.nsteps, args.discount, args.lr, args.gae_lambda,
                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                    separated_networks = args.separated_networks,
                    env_name=env_dict,
                    num_actions = ACTION_SPACE,
                    int_coef=args.intrinsic_motivation,
                    normalize_int_rewards=args.normalize_intrinsic_bonus,
                    im_type=args.im_type, 
                    int_coef_type = args.int_coef_type,
                    use_episodic_counts = args.use_episodic_counts,
                    use_only_not_visited = args.use_only_not_visited,
                    total_num_frames = args.frames,
                    # seeds to select levels
                    init_train_seed = args.init_train_seed,
                    max_num_train_seeds=args.num_train_seeds,
                    init_test_seed = args.init_test_seed,
                    max_num_test_seeds=args.num_test_seeds,
                    # OOD related
                    envs_ood=envs_ood, env_ood_name=env_dict_ood, ood_method=args.ood_method,
                    init_train_seed_ood=args.init_train_seed_ood, max_num_train_seeds_ood=args.num_train_seeds_ood,
                    init_test_seed_ood=args.init_test_seed_ood, max_num_test_seeds_ood=args.num_test_seeds_ood,
                    ood_sampling_ratio=args.ood_sampling_ratio, 
                    intervene_after=args.intervene_after, intervene_every=args.intervene_every, 
                    adaptation_strategy=args.adaptation_strategy,
                    # Visualization for testing (by saving images)
                    window_visible=bool(args.visualize_levels),
                    # Self-supervised
                    self_supervised_model=self_supervised_model, 
)
# else:
#     raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# ******************************************************************************
# Load optimizer
# ******************************************************************************

if "optimizer_state" in status:
    if separated_networks:
        algo.optimizer[0].load_state_dict(status["optimizer_state"][0])
        algo.optimizer[1].load_state_dict(status["optimizer_state"][1])
        txt_logger.info("Optimizer loaded\n")
    else:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

# ******************************************************************************
# Load SSL
# ******************************************************************************
if args.load_ssl:
    algo.optimizer_idm.load_state_dict(status["ssl_optimizer"])
    self_supervised_model.load_state_dict(status["ssl_mod_wegihts"])
    print('imported correcyl')

# ******************************************************************************
# Load Intrinsic Motivation Module
# ******************************************************************************
if args.load_im_module:
    algo.im_module.state_embedding.load_state_dict(status["im_mod_state_embedding_weights"])
    algo.im_module.optimizer_state_embedding.load_state_dict(status["im_mod_state_embedding_optimizer"])
    algo.im_module.inverse_dynamics.load_state_dict(status["im_mod_inverse_dynamics_weights"])
    algo.im_module.optimizer_inverse_dynamics.load_state_dict(status["im_mod_inverse_dynamics_optimizer"])
    algo.im_module.forward_dynamics.load_state_dict(status["im_mod_forward_dynamics_weights"])
    algo.im_module.optimizer_forward_dynamics.load_state_dict(status["im_mod_forward_dynamics_optimizer"])
    txt_logger.info("Intrinsic Motivation Module loaded\n")

    if args.ood_method in IM_MOD_METHODS:
        algo.ood_module.im_module = deepcopy(algo.im_module)  # TODO: COMPROBAR SI ESTA EN LA GRAFICA O EN LA CPU

# ******************************************************************************
# Train model
# ******************************************************************************

if args.new_dir_name:
    num_frames = 0
    update = 0
    write_headers = True
else:
    num_frames = status["num_frames"]
    update = status["update"]
    if num_frames > 0:
        write_headers = False
    elif num_frames == 0:
        write_headers = True
    else:
        raise ValueError("num_frames can't be negative")
start_time = time.time()
evaluation_ref = 0

while num_frames < args.frames:
    
    # sample current time for fps calculation
    update_start_time = time.time()

    # Create the pool of levels to sample from in the offline adaptation strategy at the beginning of the training
    #   or if we are evaluating the OOD detection performance just extract the pool of levels with OOD scores
    if (args.adaptation_strategy == 'offline' and args.env_ood != '' and num_frames == 0) or args.evaluate_ood_detection_performance:
        # Create or load centroids and thresholds
        if args.ood_method in OOD_METHODS:
            if not (args.ood_method in LOGITS_METHODS and args.evaluate_ood_detection_performance):  # To eval performance LOGITS_METHODS do not need centroids and thresholds
                #algo.ood_module.load_ood_method_info(Path(model_dir))
                if args.load_in_distribution_info:
                    txt_logger.info('*************** Loading centroids and thresholds... ***************')
                    algo.ood_module.load_ood_method_info(Path(model_dir), model_name=args.seed)
                    txt_logger.info('*************** Centroids and thresholds loaded ***************\n')
                else:
                    txt_logger.info('*************** Creating centroids and thresholds... ***************')
                    algo.generate_in_distribution_and_thresholds(tpr=args.tpr)
                    if args.save_in_distribution_info:
                        algo.ood_module.save_ood_method_info(save_path=model_dir, model_name=args.seed)  # New model dir
                        txt_logger.info('*************** In-distribution info saved ***************\n')
                
        # If we are evaluating the OOD detection performance, just extract the pool of levels with OOD scores and save them to csv
        if args.evaluate_ood_detection_performance:
            txt_logger.info('*************** Metrics for OOD detection ***************\n')
            data_per_level, data_per_step, reward_and_success_metrics = algo.generate_metrics_for_detection_performance(args.ood_detection_range)
            # Per step
            name_per_step = 'per_steps_' + args.ood_method
            save_ood_detection_results(data=data_per_step, save_path="ood_storage/detection_results/",
                                       name=name_per_step, model_name=f"{Path(model_dir).name}_{args.ood_detection_range:02d}_{args.seed:02d}")
            # Performance metrics
            save_performance_metrics(env_names=[args.env, args.env_ood], seed=args.seed, data=reward_and_success_metrics,
                                     save_path="ood_storage/", model_name=pretrained_model_dir)
            txt_logger.info('*************** OOD detection results saved ***************\n')
            txt_logger.info('Exiting program...')
            exit()

        txt_logger.info('*************** Creating pool of levels for offline adaptation strategy... ***************')
        algo.create_pool_of_levels()
        txt_logger.info('*************** Pool of levels created ***************')

    # If we are using OOD envs
    # Only eval if it is the first time or if we have reached the evaluation interval
    if (len(env_dict_ood) > 0 and num_frames == 0) or ((args.num_test_seeds>0) and (num_frames > evaluation_ref + args.evaluation_interval)):
        evaluation_ref = num_frames
        txt_logger.info('*************** Evaluating agent in mixed environments... ***************')
        ev_in_distr_successrate, ev_avgsteps, ev_avgret = algo.evaluate_agent(strategy='level_ids_in_order', eval_in_ood_envs=False)
        ev_ood_successrate, ev_ood_avgsteps, ev_ood_avgret = algo.evaluate_agent(strategy='level_ids_in_order', eval_in_ood_envs=True)
        # Compute the real values of the metrics, weighting by the sampling ratio of levels that are normal/ood
        ev_real_successrate = ev_in_distr_successrate * (1-args.ood_sampling_ratio) + ev_ood_successrate * args.ood_sampling_ratio
        ev_real_avgsteps = ev_avgsteps * (1-args.ood_sampling_ratio) + ev_ood_avgsteps * args.ood_sampling_ratio
        ev_real_avgret = ev_avgret * (1-args.ood_sampling_ratio) + ev_ood_avgret * args.ood_sampling_ratio

        # Log evaluation metrics (csv and tensorboard only)
        header_ood = ["update", "frames", "ev_real_success_rate", "ev_success_rate", "ev_ood_success_rate", "ev_real_avg_return", "ev_avg_return", "ev_ood_avg_return", "ev_real_avg_steps", "ev_avg_steps", "ev_ood_avg_steps",]        
        data_ood = [update, num_frames, ev_real_successrate, ev_in_distr_successrate, ev_ood_successrate, ev_real_avgret, ev_avgret, ev_ood_avgret, ev_real_avgsteps, ev_avgsteps, ev_ood_avgsteps]
        txt_logger.info("U {} | F {:06} | SR_real {:.2f} | SR_in_distr {:.2f} | SR_ood {:.2f} | AR_real {:.2f} | AR_in_distr {:.2f} | AR_ood {:.2f} | AS_real {:.2f} | AS_in_distr {:.2f} | AS_ood {:.2f}".format(*data_ood))
        #header_ood = ["update", "frames", "ev_success_rate", "ev_avg_steps", "ev_avg_return", "ev_ood_success_rate", "ev_ood_avg_steps", "ev_ood_avg_return", "ev_real_success_rate", "ev_real_avg_steps", "ev_real_avg_return"]
        #data_ood = [update, num_frames, ev_in_distr_successrate, ev_avgsteps, ev_avgret, ev_ood_successrate, ev_ood_avgsteps, ev_ood_avgret, ev_real_successrate, ev_real_avgsteps, ev_real_avgret]
        #txt_logger.info("U {} | F {:06} | SR {:.2f} | SR_ood {:.2f} | SR_real {:.2f} | AS {:.2f} | AS_ood {:.2f} | AS_real {:.2f} | AR {:.2f} | AR_ood {:.2f} | AR_real {:.2f}".format(*data_ood))
        if status["num_frames"] == 0:
            csv_ood_logger.writerow(header_ood)
        csv_ood_logger.writerow(data_ood)
        csv_ood_file.flush()
        for field, value in zip(header_ood, data_ood):
            tb_writer.add_scalar(field, value, num_frames)
        txt_logger.info('*************** Evaluation completed ***************')
    
    # Data collection
    exps, logs1 = algo.collect_experiences()
    # PPO update
    if args.use_only_ssl==0:
        logs2 = algo.update_parameters(exps)
    else:
        logs2 = {
            "entropy": 0,
            "policy_loss": 0,
            "value": 0,
            "value_loss": 0,
            "grad_norm": 0,
            "grad_norm_critic": 0
        }
    
    # SELF-SUPERVISON LOSS
    if args.use_ssl != 0 and update % args.ssl_update_freq == 0:
        logs3 = algo.update_ssl_with_idm(exps)
    else:
        logs3 = {"ssl_loss":0}
            
    # Logging parameters
    logs = {**logs1, **logs2, **logs3}
    update_end_time = time.time()
    num_frames += logs["num_frames"]
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        episodes = logs["episode_counter"]

        # intrinsic
        return_int_per_episode = utils.synthesize(logs["return_int_per_episode"])
        return_int__norm_per_episode = utils.synthesize(logs["return_int_per_episode_norm"])
        # extrinsic
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])


        # general values
        header = ["update", "frames", "FPS", "duration","episodes"]
        data = [update, num_frames, fps, duration, episodes]
        only_txt = [update, num_frames, fps, duration, episodes]

        # add beta coef
        header += ["weight_int_coef"]
        data += [logs["weight_int_coef"]]
        only_txt += [logs["weight_int_coef"]]

        # returns
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        only_txt += [rreturn_per_episode["mean"]]
        only_txt += [rreturn_per_episode["std"]]

        header += ["return_int_" + key for key in return_int_per_episode.keys()]
        data += return_int_per_episode.values()
        only_txt += [return_int_per_episode["mean"]]
        only_txt += [return_int_per_episode["std"]]

        # avg 100 episodes
        header += ["avg_success","avg_return"]
        data += [logs["avg_success"],logs["avg_return"]]
        only_txt += [logs["avg_success"]]
        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm", "grad_norm_critic","ssl_loss"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["grad_norm_critic"], logs["ssl_loss"]]
        only_txt += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["grad_norm_critic"], logs["ssl_loss"]]

        header+= ["hist_ret_avg","normalization_int_score","predominance_ext_over_int"]
        data += [logs["hist_ret_avg"],logs["normalization_int_score"],logs["predominance_ext_over_int"]]

        # OOD envs detected
        # if args.env_ood:
        #     header += ["number_of_ood_levels_detected", "number_of_in_distr_levels_detected"]
        #     data += [logs["number_of_ood_levels_detected"], logs["number_of_in_distr_levels_detected"]]

        # # Metrics in EV environments
        # if args.num_test_seeds > 0:
        #     header += ["ev_success_rate","ev_avg_steps","ev_avg_return"]
        #     data += [ev_in_distr_successrate,ev_avgsteps,ev_avgret]
        # if len(env_dict_ood) > 0:
        #     header += ["ev_ood_success_rate","ev_ood_avg_steps","ev_ood_avg_return"]
        #     data += [ev_ood_successrate,ev_ood_avgsteps,ev_ood_avgret]
        # if args.num_test_seeds_ood > 0 and len(env_dict_ood) > 0:
        #     header += ["ev_real_success_rate","ev_real_avg_steps","ev_real_avg_return"]
        #     data += [ev_real_successrate, ev_real_avgsteps,ev_real_avgret]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | Eps {} | β:{:.5f} | rR:μσ {:.2f} {:.2f} | rRi:μσ {:.2f} {:.2f} | SR: {:.2f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇p {:.3f} | ∇c {:.3f} | sslL {:.3f}"
            .format(*only_txt))
        # txt_logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | H {:.3f} | V {:.3f} | Ve {:.3f} | Vi {:.3f}  | pL {:.3f} | vL {:.3f} | vLe {:.3f} | vLi {:.3f} | ∇p {:.3f} | ∇c {:.3f}"
        #     .format(*data))
        # with normalized intrinsic return
        # txt_logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_intNORM:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | H {:.3f} | V {:.3f} | Ve {:.3f} | Vi {:.3f}  | pL {:.3f} | vL {:.3f} | veL {:.3f} | viL {:.3f} | ∇ {:.3f}"
        #     .format(*data))

        # THIS ONLY NECESSARY IF MODIFICATED RESHAPE REWARD
        # header += ["return_" + key for key in return_per_episode.keys()]
        # data += return_per_episode.values()

        if status["num_frames"] == 0 or write_headers:
            csv_logger.writerow(header)
            write_headers = False
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:

        # Get weights of AC
        if separated_networks:
            acmodel_weights = (acmodel[0].state_dict(),
                                acmodel[1].state_dict())
            optimizer_state = (algo.optimizer[0].state_dict(),
                                algo.optimizer[1].state_dict())
        else:
            acmodel_weights = acmodel.state_dict()
            optimizer_state = algo.optimizer.state_dict()

        # set dictionary used to save
        status = {"num_frames": num_frames, "update": update,
                    "model_state": acmodel_weights, "optimizer_state": optimizer_state,}
        
        if self_supervised_model is not None:
            # ssl model
            ssl_model_weights = self_supervised_model.state_dict()
            ssl_optimizer = algo.optimizer_idm.state_dict()
            save_more_params = {"ssl_mod_wegihts": ssl_model_weights,
                                "ssl_optimizer": ssl_optimizer
            }
            status.update(save_more_params)
            
        if args.save_im_module:
            im_mod_state_embedding_weights = algo.im_module.state_embedding.state_dict()
            im_mod_state_embedding_optimizer = algo.im_module.optimizer_state_embedding.state_dict()
            im_mod_inverse_dynamics_weights = algo.im_module.inverse_dynamics.state_dict()
            im_mod_inverse_dynamics_optimizer = algo.im_module.optimizer_inverse_dynamics.state_dict()
            im_mod_forward_dynamics_weights = algo.im_module.forward_dynamics.state_dict()
            im_mod_forward_dynamics_optimizer = algo.im_module.optimizer_forward_dynamics.state_dict()
            
            save_more_params = {
                    "im_mod_state_embedding_weights": im_mod_state_embedding_weights,
                    "im_mod_state_embedding_optimizer": im_mod_state_embedding_optimizer,                    
                    "im_mod_inverse_dynamics_weights": im_mod_inverse_dynamics_weights,
                    "im_mod_inverse_dynamics_optimizer": im_mod_inverse_dynamics_optimizer,
                    "im_mod_forward_dynamics_weights": im_mod_forward_dynamics_weights,
                    "im_mod_forward_dynamics_optimizer": im_mod_forward_dynamics_optimizer
            }
            status.update(save_more_params)
        # else:
        #     status = {"num_frames": num_frames, "update": update,
        #             "model_state": acmodel_weights, "optimizer_state": optimizer_state,}
            
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
