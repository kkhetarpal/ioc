# Options of Interest: Temporal Abstraction with Interest Functions
This repo contains code accompaning the paper, Options of Interest: Temporal Abstraction with Interest Functions (Khetarpal et al., AAAI 2020). It includes code for **interest-option-critic (IOC)** to run all the experiments described in the paper.


* You can find demonstrative vidoes of the trained agents on our [project webpage](https://sites.google.com/view/optionsofinterest).
* All proofs, pseudo-code and reproducibility checklist details are available in the appendix on our [project webpage](https://sites.google.com/view/optionsofinterest).
* For experiment details, please refer to the full paper provided on the webpage. 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Tabular Experiments (Four-Rooms)

##### Dependencies
To install dependencies for control experiments: run the following commands:
```
conda create -n interest python=3.6
conda actvate interest
pip install seaborn
pip install matplotlib
```

##### Usage
To run the ioc code, use:
```
python interestoptioncritic_tabular_fr.py --baseline --discount=0.99 --epsilon=0.01 --noptions=4 --lr_critic=0.5 --lr_intra=0.25 --lr_term=0.25 --lr_interestfn=0.15 --nruns=10 --nsteps=2000 --nepisodes=500 --seed=7200
```

To run the baseline oc code, use:
```
python optioncritic_tabular_fr.py --baseline --discount=0.99 --epsilon=0.01 --noptions=4 --lr_critic=0.5 --lr_intra=0.25 --lr_term=0.25 --nruns=10 --nsteps=2000 --nepisodes=500 --seed=7200
```

##### Performance and Visualizations
To visualize the environment itself, use the notebook: `fr_env_plots.ipynb`

To plot the performance curves, use the notebook: `fr_analysis_performance.ipynb`

To visualize the options learned, use the notebook: `fr_analysis_heatmaps.ipynb`

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Control Experiments (TMaze & HalfCheetah)


##### Dependencies
To install dependencies for control experiments: run the following commands:
```
conda create -n intfc python=3.6
conda actvate intfc
pip install tensorflow
pip install -e . (in the main directory)
pip install gym==0.9.3
pip install mujoco-py==0.5.1
brew install mpich
pip install mpi4py
```


##### Usage
To run the code with TMaze experiments, use:
```python run_mujoco.py --env TMaze  --opt 2 --seed 2 --switch```


To run the code with HalfCheetah experiments, use:
```python run_mujoco.py --env HalfCheetahDir-v1  --opt 2 --seed 2 --switch```


##### Running experiments on slurm
To run the code on compute canada or any slurm cluster, make sure you have installed all dependencies and created a conda environment _intf_, following which use the script launcher_miniworld.sh by running:
```
chmod +x launcher_control.sh
./launcher_control.sh
```

To run the baseline option-critic, use the flag `--nointfc` in the above script:
```
k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" python run_mujoco.py --env "$envname" --saves --opt 2 --seed ${_seed} --mainlr ${_mainlr} --piolr ${_piolr} --switch --nointfc --wsaves"
```

##### Performance and Visualizations
To plot the learning curves, use the script: `control/baselines/ppoc_int/plot_res.py` with appropiate settings. 

To load and run a trained agent, use:
```
python run_mujoco.py --env HalfCheetahDir-v1 --epoch 400 --seed 0
``` 
where _epoch_ would be the training epoch at which you want to visualize the learned agent. This assumes that the saved model directory is in the ppoc_int folder.



------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Visual Navigation Experiments (Miniworld)

##### Dependencies
To install dependencies for miniworld experiments: run the following commands:
```
conda create -n intfc python=3.6
conda actvate intfc
pip install tensorflow
pip install -e . (in first directory of baselines)
brew install mpich
pip install mpi4py
pip install matplotlib
# to run the code with miniworld
pip install gym==0.10.5
```

To install [miniworld](https://github.com/maximecb/gym-miniworld): follow these [installation instructions](https://github.com/maximecb/gym-miniworld#installation).

Since the cnn policy code is much slower than mujoco experiments, the optimal way to run is using a cluster. To run miniworld headless and training on a cluster, follow these instructions [here](https://github.com/maximecb/gym-miniworld/blob/master/docs/troubleshooting.md#running-headless-and-training-on-aws).


##### Usage
To run the code headless for oneroom task with transfer, use:
```
xvfb-run -n 4005 -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python run_miniw.py --env MiniWorld-OneRoom-v0 --seed 5 --opt 2 --saves --mainlr 1e-4 --intlr 9e-5 --switch --wsaves
```


##### Running experiments on slurm
To run the code on compute canada or any slurm cluster, make sure you have installed all dependencies and created a conda environment _intf_, following which use the script launcher_miniworld.sh by running:
```
chmod +x launcher_miniworld.sh
./launcher_miniworld.sh
```

Please note that to ensure that miniworld code runs correctly headless, we here make sure we specify an exclusive port per run. 
If the port# overlaps for multiple jobs, the jobs will fail. Ideally there has to be a better way to do this, but this is the one we found easiest to make it work. Depending on how many jobs you want to launch (e.x. runs/seeds), set the range for port accordingly.


To run the baseline option-critic, use the flag `--nointfc` in the above script in the run command.



##### Performance and Visualizations
To plot the learning curves, use the script: `miniworld/baselines/ppoc_int/plot_res.py` with appropiate settings. 

To visualize the trajectories of trained agents: make the following changes in your local installation of the miniworld environment code: https://github.com/kkhetarpal/gym-miniworld/commits/master
Load and run the trained agent to visualize the trajectory of the trained agents with a 2-D top-view of the 3D oneroom.

To load and run a trained agent, use:
```
python run_miniw.py --env MiniWorld-OneRoom-v0 --epoch 480 --seed 0
``` 
where _epoch_ would be the training epoch at which you want to visualize the learned agent. This assumes that the saved model directory is in the ppoc_int folder.



### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/kkhetarpal/ioc/issues).


### Additional Material

* Poster presented at NeurIPS 2019, Deep RL Workshop, Learning Transferable Skills Workshop. ([link](https://kkhetarpal.files.wordpress.com/2019/12/neurips_drl_optionsofinterest_poster.pdf))
* Preliminary ideas presented in AAAI 2019, Student Abstract track, Selected as a finalist in 3MT Thesis Competition ([paper link](https://www.aaai.org/ojs/index.php/AAAI/article/view/5114)), ([poster link](([link](https://kkhetarpal.files.wordpress.com/2019/08/poster_interestfunctions.pdf))).


### Citations
* The fourrooms experiment is built on the Option-Critic, [2017](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14858/14328) [tabular code](https://github.com/jeanharb/option_critic/tree/master/fourrooms).
* The PPOC, [2017](https://arxiv.org/pdf/1712.00004.pdf) baselines [code](https://github.com/mklissa/PPOC) serves as base to our function approximation experiments.
* To install Mujoco, please visit their [website](https://www.roboti.us/license.html) and acquire a free student license.
* For any issues you face with setting up miniworld, please visit their [troubleshooting](https://github.com/maximecb/gym-miniworld/blob/master/docs/troubleshooting.md) page.

