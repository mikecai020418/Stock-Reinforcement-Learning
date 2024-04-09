import pandas as pd
import  matplotlib
import matplotlib.pyplot as plt
from dqn_agent import Agent
import torch
import numpy as np
from main import getState


STATE_SIZE = 10

# df = pd.read_csv('CSV/AAPL_evaluate.csv')
# df = pd.read_csv('CSV/PG_evaluate.csv')
df = pd.read_csv('CSV/TSLA_evaluate.csv')


print(df['Close'])
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)

agent = Agent(state_size=STATE_SIZE, action_size=3)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

stockData = list(df['Close'])
l = len(stockData)-1
window_size = 10
state = getState(stockData, 0, window_size + 1)
total_profit = 0
agent.inventory = []
action_list = []
value_list = []
for t in range(l):
    action = agent.act(state, eps=0)
    next_state = getState(stockData, t + 1, STATE_SIZE + 1)
    if action == 1:# Buy
        agent.inventory.append(stockData[t])
        action_list.append(action)
            # print("buy" + str(stockData[t]))
    elif action == 2 and len(agent.inventory) > 0: # Sell
        bought_price = agent.inventory.pop(0)
        total_profit += stockData[t] - bought_price
        action_list.append(action)
    else:
        action_list.append(0)

    done = 1 if t == l - 1 else 0
    state = next_state
    value_list.append(stockData[t])
    if done:
        print("------------------------------")
        print("total_profit = " + str(total_profit))
        print("------------------------------")
        #plt.plot(np.arange(len(value_list)), value_list)
        action_list.append(0)
        df['action'] = pd.DataFrame(action_list).values
        df["Close"].plot(figsize=(8, 5), grid=True)
        sell = (df['action'].values == 2)
        plt.scatter(df.index[sell], df["Close"].values[sell], c='r')
        buy = (df['action'].values == 1)
        plt.scatter(df.index[buy], df["Close"].values[buy], c='b')
        plt.legend(['value', 'sell', 'buy'])
        plt.title('total profit %f'%(total_profit))
        plt.savefig('evaluating_graph.png')
        plt.show()