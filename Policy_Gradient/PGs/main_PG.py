import torch
import gym
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from CartPole.Policy_Gradient.agent_PG import Agent_PG
from CartPole.Policy_Gradient.PPO_with_R import PPO_v1
from CartPole.Policy_Gradient.PPO_with_A import PPO_V2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_file="models/pg_model_3.pth"
# plot_file="results&plots/pg_3.png"


def watch_smart_agent(agent,model_name):
    agent.policy.load_state_dict(torch.load(model_name))
    state = env.reset()
    for t in range(1000):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action,_ = agent.policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            print("done in time step {}".format(t+1))
            break
    env.close()


def plot_scores(scores,file_name,multi_time=False):
    "绘制多次训练多条曲线"
    if multi_time:
        x=np.arange(1, len(scores[0]) + 1)
        for n in range(len(scores)):
            rolling_mean = pd.Series(scores[n]).rolling(100).mean()
            plt.plot(x,rolling_mean,label="trial_"+str(n+1))
    else:
        x = np.arange(1, len(scores) + 1)
        rolling_mean = pd.Series(scores).rolling(100).mean()
        plt.plot(x, rolling_mean)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def plot_diff_agent(scores_2d,file_name):
    " 绘制多种不同agent的训练曲线：多曲线图"
    for name,scores in scores_2d:
        x = np.arange(1, len(scores) + 1)
        rolling_mean = pd.Series(scores).rolling(100).mean()
        plt.plot(x, rolling_mean,label=name)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def agent_test(agent,n_episode,model_name):
    agent.policy.load_state_dict(torch.load(model_name))
    scores = []
    for i_episode in range(1, n_episode + 1):
        rewards=[]
        state = env.reset()
        while True:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            action, _ = agent.policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores.append(sum(rewards))

    return scores


def train_agent(env,agent,n_episode,model_file):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episode + 1):
        total_reward=agent.train(env)
        # record scores(total rewards) per episode
        scores_deque.append(total_reward)
        scores.append(total_reward)

        print('\r Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
              .format(i_episode, np.mean(scores_deque), total_reward), end="")
        if i_episode % 100 == 0:
            print('\n Episode {}\t Average Score: {:.2f}\n'.format(i_episode,np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}\n----------\n'.format(i_episode,
                                                                                       np.mean(scores_deque)))
            torch.save(agent.policy.state_dict(),model_file)
            break

    return scores


def train_agent_multi_times(env, agent, n_episode, train_time, file):
    " 一个 agent 训练多次并绘制所有的奖励曲线，考察特定 policy gradient 算法的稳定性"
    scores_2d = []
    for n in range(train_time):
        scores = []
        for i_episode in range(1, n_episode + 1):
            total_reward = agent.train(env)
            scores.append(total_reward)

        print('Trial {} finished. \t Avg score for the last 100 episode: {}'
              .format((n + 1), np.mean(scores[-100:])))
        scores_2d.append(scores)

    plot_scores(scores_2d, file,multi_time=True)


def train_diff_agents(env,agents,n_episode,file):
    " 训练多种算法的不同agent, 绘制奖励曲线对比性能 "
    scores_2d=[]
    for name in agents.keys():
        scores = []
        for i_episode in range(1, n_episode + 1):
            total_reward = agents[name].train(env)
            scores.append(total_reward)
        scores_2d.append((name,scores))
        print('Training agent {} finished. \t Avg score for the last 100 episode: {}'\
            .format(name,np.mean(scores[-100:])))

    plot_diff_agent(scores_2d,file)


if __name__=="__main__":
    env = gym.make('CartPole-v0')

    agent_pg = Agent_PG(state_size=4,action_size=2,type="pg")
    agent_rf=Agent_PG(state_size=4,action_size=2,type="reinforce")
    ppo_R=PPO_v1(state_size=4,action_size=2)

    ppo_without_entropy=PPO_V2(state_size=4,action_size=2,add_entropy=False)
    ppo_with_entropy=PPO_V2(state_size=4,action_size=2,add_entropy=True)

    #train_scores = train_agent(env, ppo_with_entropy, 2000, 'PGs/models/PPO_new.pth')
    #plot_scores(train_scores, 'PGs/results&plots/PPO_with_entropy_1.png')

    # agents={'PPO with R':ppo_R,
    #         'PPO with A':ppo_with_entropy,
    #         'Policy Gradient':agent_pg,
    #         'Reinforce':agent_rf}

    ppo_agents={'PPO_R':ppo_R,'PPO_A_org':ppo_without_entropy,'PPO_A_entropy':ppo_with_entropy}

    train_diff_agents(env, ppo_agents, 1500, '../results&plots/PPO_comparison_4.png')
    # train_agent_multi_times(env,ppo_with_entropy,1300,5,'PGs/results&plots/PPO-entropy_5times.png')










