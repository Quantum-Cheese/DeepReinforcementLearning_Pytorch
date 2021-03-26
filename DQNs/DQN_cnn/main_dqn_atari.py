import argparse
import os
import arrow
import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from dqn_agent import Agent_dqn
import atari_wappers

parser = argparse.ArgumentParser()
'''------------------###### 基本参数 ######--------------------'''
parser.add_argument('env_name',type=str)
parser.add_argument('run_mode',type=str,help='运行模式:train/test')
'''------------------###### 测试参数 #####--------------------'''
parser.add_argument('--test_episode',type=int,default=500,help='测试的episode数量',required=False)
parser.add_argument('--test_model_file',type=str,default='',help = '测试用的模型文件路径',required=False)
parser.add_argument('--test_video_play',type=str,default='no',
                    help='测试时是否需要显示游戏运行界面(默认都保存得分曲线) yes/no',required=False)
'''------------------###### 训练参数 #####--------------------'''
parser.add_argument('--train_episode',type=int,default=2000,help='训练的episode数量',required=False)
parser.add_argument('--learning_rate',type=float,default=5e-3,help='优化器的学习率',required=False)
parser.add_argument('--buffer_size',type=int,default=10000,help='Replay buffer 的最大容量',required=False)
parser.add_argument('--batch_size',type=int,default=32,help='每次训练的 batch 大小',required=False)
parser.add_argument('--gamma',type=float,default=0.99,help='折扣率',required=False)
parser.add_argument('--update_every',type=int,default=5,help='每隔多少 time step 训练一次',required=False)
parser.add_argument('--eps_decay',type=float,default=0.995,help='epsilon 的衰减率',required=False)
args = parser.parse_args()


def plot_scores(scores,filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)


def train_agent(env,agent,n_episode,eps_decay,gamma,update_every):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    start_time = arrow.now()
    for i_episode in range(1, n_episode + 1):
        state = env.reset()
        score = 0
        episode_loss=[]
        while True:
            # get action
            action = agent.act(state,i_episode,eps_decay)
            # interact with env (one step)
            next_state, reward, done, _ = env.step(action)
            # train the agent
            sarsd = (state, action, reward, next_state,done)
            loss = agent.step(sarsd,gamma,update_every)
            # update status
            state = next_state
            score += reward
            # break the loop if current episode is over
            if done:
                break
            if loss is not None:
                episode_loss.append(loss)

        # update rewards and scores every episode
        scores_window.append(score)
        scores.append(score)

        print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode, np.mean(episode_loss),
                                                                       np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode, np.mean(episode_loss),
                                                                           np.mean(scores_window)))
            print('\rRunning time till now :{}\n'.format(arrow.now() - start_time))

            path = 'Models/'
            if not os.path.exists(path):
                os.makedirs(path)
            model_file = 'Models/CNN_model' + '|' + arrow.now().format('MM-DD#HH:mm')
            torch.save(agent.qnetwork_local.state_dict(), model_file)

    return scores


def get_agent_scores(env,agent,n_episode):
    total_rewards = []
    for _ in range(0,n_episode):
        state = env.reset()
        episode_score = 0
        while True:
            if args.test_video_play == 'yes':
                env.render()
            action = agent.act_greedy_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_score += reward
            if done:
                break
        total_rewards.append(episode_score)
    return total_rewards


def trained_agent_test(env_name,model_file,n_episode,test_score_file):
    env = atari_wappers.make_env(env_name)
    state_size, action_size = env.observation_space.shape, env.action_space.n

    agent = Agent_dqn(state_size, action_size)
    agent.qnetwork_local.load_state_dict(torch.load(model_file))

    test_score = get_agent_scores(env,agent,n_episode)
    plot_scores(test_score,test_score_file)


'''
--------------------------
### 命令行启动（参数设置）###
--------------------------
'''
if args.run_mode == 'test':
    print('####################################################\n'
          ' Start Testing trained DQN_CNN agent on {} environment.\n'
          '####################################################\n'.format(args.env_name))
else:
    print('####################################################\n'
          ' Start Training on {} environment using DQN with CNN\n'
          '####################################################\n'.format(args.env_name))

if args.run_mode =='test':
    print(args.env_name, args.run_mode, args.test_video_play, args.test_episode, args.test_model_file)

    path = 'Plots/'
    if not os.path.exists(path):
        os.makedirs(path)
    test_score_file = 'Plots/test-score' + '|' + arrow.now().format('MM-DD#HH:mm')+'.png'
    # test trained agent
    trained_agent_test(args.env_name, args.test_model_file, args.test_episode, test_score_file)
else:
    print('Training Parameters :\n Train episode : {}\n Network update every {} time step \n '
          'Replay buffer size : {}\n Batch size : {}\n '
          'Learning rate : {} \n GAMMA : {} \n Epsilon decay rate : {}\n'
          .format(args.train_episode,args.update_every,args.buffer_size,args.batch_size,args.learning_rate,args.gamma,args.eps_decay))
    env = atari_wappers.make_env(args.env_name)

    state_size, action_size = env.observation_space.shape, env.action_space.n
    agent = Agent_dqn(state_size, action_size,args.learning_rate,args.buffer_size,args.batch_size)
    # start training
    trained_scores = train_agent(env, agent, args.train_episode,args.eps_decay, args.gamma, args.update_every)

    path = 'Plots/'
    if not os.path.exists(path):
        os.makedirs(path)
    train_score_file = 'Plots/train-score' + '|' + arrow.now().format('MM-DD#HH:mm') + '.png'
    plot_scores(trained_scores, train_score_file)


# if __name__ =="__main__":
#     env = gym.make('SpaceInvaders-v0')
#     state_size, action_size = env.observation_space.shape, env.action_space.n
#     dqn_agent = Agent_dqn(state_size,action_size)
#     train_agent(env, dqn_agent, 1000, "Models/dqnCNN_model_0324.pth")
