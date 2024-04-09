# Out Of Distribution techniques for a better adaptation and generalization in Reinforcement Learning

## Installation

1. Clone this repository (tested with **Python 3.10**).

2. Install the dependencies with *pip3*, so that `gym-minigrid`(environment), `pytorch` and other necessary packages/libraries are installed:

```
pip3 install -r requirements.txt
```
**Note:** This code was accordingly modified from: https://github.com/lcswillems/rl-starter-files. The `torch_ac` holds the same structure but does not work as the original implementation. Nevertheless, the example of usage is almost straightforward.  

## Reproduce results

To reproduce the results, run the experiments on simulation scripts folder in this order:

__*IMPORTANT NOTE:*__ The GPU ids are arbitrary, so got into the ```.sh``` scripts and to the end of the line to change those as ```--gpu-id <gpu_id_wanted>``` 

1. ```MN5S8_train.sh```
2. ```MN5S8_train_with_ssl.sh```
3. All RQ1 scripts
4. All RQ2 scripts


## Example of use
In the `simulation_scripts` folder are provided the necessary scripts to train the agent with some intrinsic motivation techniques. Use either 1st or episdodic ones for better results.

### Training 
 An example of a single simulation is as follows:
```
python3 -m scripts.train --model KS3R3_c_im0005_ent00005_1 --seed 1  --save-interval 10 --frames 30000000  --env 'MiniGrid-KeyCorridorS3R3-v0' --intrinsic-motivation 0.005 --im-type 'counts' --entropy-coef 0.0005 --normalize-intrinsic-bonus 0 --separated-networks 0
```

The hyperparameters and different criterias can me modified by reference them directly from the command line (or by modifying the default values directly in the `scripts/train.py`). Some of the most important hyperparameters are:
*   `--env`: environment to be used
*   `--frames`: the number of frames/timesteps to be run
*   `--seed`: the seed used to reproduce results
*   `--im_type`: specifies which kind of intrinsic motivation module is going to be used to compute the intrinsic reward
*   `--intrinsic_motivation`: the intrinsic coefficient value
*   `--separated-networks`: used to determine if the actor-critic agent will be trained with a single two-head CNN architecture or with two independent networks
*   `--model`: the directory where the logs and the models are saved

For example, setting the `--intrinsic_motivation 0` means that the agent will be trained without intrinsic rewards.

### Evaluation
#### Example 1 (using the default environments)
```
python3 -m scripts.evaluate --model MN7S8_c_1st_im0005_ent00005_1 --env MiniGrid-MultiRoom-N7-S8-v0 --num_episodes 10
```
Evaluates the trained model *MN7S8_c_1st_im0005_ent00005_1* over the environment *MiniGrid-MultiRoom-N7-S8-v0* with 10 different random seeds.


#### Example 2 (pre-loaded numpy environment/maze)
By the virute of using:
*   `--env-list`: loads an environment/maze that is codified in .npy format. The file has to be stored at /numpyworldfiles. It does not require to use `--env`.

```
python3 -m scripts.evaluate --model MN3S10_c_1st_im0005_ent00005_1 --env-list MN3S38_test_lava
```

