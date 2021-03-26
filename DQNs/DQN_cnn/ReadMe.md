
### To start the project

#### 0. Basic settings
* `env_name`: (str) name of the gym atari env that you want to play with 
* `run_mode` : (str:train/test) 

#### 1. To train an agent using DDQN with CNN network 
* `train_episode`
* `learning_rate`
* `buffer_size`
* `batch_size`
* `gamma`
* `update_every`
* `eps_decay`

exp. 
``` 
python main_dqn_atari.py SpaceInvaders-v0 train --learning_rate 1e-3
```

To run in back ground and save a log file:
```
nohup python -u main_dqn_atari.py SpaceInvaders-v0 train --learning_rate 1e-3 > train_20210326.log 2>&1 &
```

#### 2. To test a trained agent
* `test_episode` (int) number of episodes you what to test the agent
* `test_model_file` (str) path of the model file corresponding with the trained agent you want to test 
* `test_video_play` (str:yes/no) whither you want to watch video playing during testing 

exp.
```
python main_dqn_atari.py SpaceInvaders-v0 test --test_episode 500 --test_model_file Models/dqnCNN_model_0324.pth --test_video_play no
```

