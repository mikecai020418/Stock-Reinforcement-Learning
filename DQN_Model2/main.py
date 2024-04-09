from dqn_agent import Agent
from model import QNetwork
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

STATE_SIZE = 10
EPISODE_COUNT = 1000

def dqn(n_episodes=EPISODE_COUNT, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    episode_info_list = []
    for i_episode in range(1, n_episodes+1):
        print("Episode" + str(i_episode))
        state = getState(stockData, 0, STATE_SIZE + 1)
        total_profit = 0
        agent.inventory = []
        total_buy_actions = 0
        total_sell_actions = 0
        eps = eps_start

        for t in range(l):
            action = agent.act(state, eps)
            next_state = getState(stockData, t + 1, STATE_SIZE + 1)
            reward = 0

            if action == 1:# Buy
                agent.inventory.append(stockData[t])
                total_buy_actions += 1
                # print("buy" + str(stockData[t]))
            elif action == 2 and len(agent.inventory) > 0: # Sell
                bought_price = agent.inventory.pop(0)
                total_profit += stockData[t] - bought_price
                # reward = max(stockData[t] - bought_price, 0)
                reward = stockData[t] - bought_price
                total_sell_actions += 1
                # print("Sell: " + str(stockData[t]) + " | Profit: " + str(stockData[t] - bought_price))
            done = 1 if t == l - 1 else 0
            agent.step(state, action, reward, next_state, done)
            eps = max(eps_end, eps * eps_decay)
            state = next_state

            if done:
                episode_info_list.append({
                    'Episode': i_episode,
                    'Total_Buy_Actions': total_buy_actions,
                    'Total_Sell_Actions': total_sell_actions
                })

                print("------------------------------")
                print("total_profit = " + str(total_profit))
                print("------------------------------")

        scores.append(total_profit)
    episode_df = pd.DataFrame(episode_info_list)
    print(episode_df)
    episode_df.to_excel('episode_info.xlsx', index=False, engine='openpyxl')

    return scores


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


if __name__ == '__main__':
    stockData = []
    # df = pd.read_csv('CSV/AAPL.csv')
    # df = pd.read_csv('CSV/PG.csv')
    df = pd.read_csv('CSV/TSLA.csv')

    stockData = list(df['Close'])

    agent = Agent(state_size=STATE_SIZE, action_size=3)
    l = len(stockData) - 1

    scores = dqn()
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('training_scores.png')
    plt.show()