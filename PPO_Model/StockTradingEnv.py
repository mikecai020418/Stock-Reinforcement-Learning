import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import tensorflow as tf


#Bookkeeping
MAX_ACCOUNT_BALANCE=2000000
MAX_SHARE_PRICE=50000
MAX_NUM_SHARES=2000000
INITIAL_ACCOUNT_BALANCE=10000
MAX_STEPS=30


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.transactions = []  #For recording
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(   # This is not actually right as this is a cts space and tfp.distributions.Categorical is a discrete space
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_date - 6 : self.current_date -1, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_date - 6: self.current_date -1, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_date - 6: self.current_date -1, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_date - 6: self.current_date -1, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_date - 6: self.current_date -1, 'Volume'].values / MAX_NUM_SHARES,
            self.df.loc[self.current_date - 6: self.current_date -1, 'VWAP'].values/1069.5784397582636 ,
            self.df.loc[self.current_date - 6: self.current_date -1, 'RSI'].values/96.08478275173084 
        ])

        # Append additional data and scale each value to between 0-1
        # print(self.current_step)
        # print(frame.shape)
        obs1 = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        obs =obs1
        return obs

    def _take_action(self, action_prob_val):
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_date, "Open"], self.df.loc[self.current_date, "Close"])
        action_type = int(action_prob_val[0])
    
        #amount= float(probs_list[action_type])
        
        amount=0.1
        #amount = float(action_prob_val[1])  #by certainty 
        #amount = (1-action_prob_val[1]/(-4))  #This is the percentage of stocks that shall be traded. -10 is an attempt to normalize the result to be [0,1], considering p<0.0001
        #amount = tf.sigmoid(-action_prob_val[1])  # Sigmoid ensures output is in [0,1]
        #amount=0.        #print("amount",amount)
        #print("amount",amount)

        shares_bought=0
        shares_sold=0
        total_possible = int(self.balance / current_price)

        current_date = self.df.loc[self.current_date, "Date"]  #For recording
        current_close_price = self.df.loc[self.current_date, "Close"]

        #amount = float((1-action_prob_val[1]/(-4)))
        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price #Trading cost
            #print("ttl",total_possible,"shares b",shares_bought)
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought+1e-6)
            self.shares_held += shares_bought
            self.transactions.append((current_date, current_close_price, 'buy', shares_bought)) #viz
        
        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price  #Trading cost
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price
            self.transactions.append((current_date, current_close_price, 'sell', shares_sold)) #viz

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

        #print("Current shares:", self.shares_held, "  Total Possible:", total_possible, "action: ",shares_bought,"/", shares_sold)

    def step(self, action_prob_val):
        # Execute one time step within the environment
        self._take_action(action_prob_val)

        self.current_date += 1
        self.current_step +=1
        done=False
        if self.current_date > len(self.df.loc[:, 'Open'].values) - 1:
            self.current_date = 6
            #done=True
            #print("done!, with self.net_worth",self.net_worth,"current step",self.current_step )

        delay_modifier = (self.current_step / 300)

        reward = self.balance * delay_modifier
        done = ((self.net_worth <= 0)or(self.current_step>MAX_STEPS)or(self.net_worth>2*INITIAL_ACCOUNT_BALANCE))
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.pnl = 0
        self.transactions = []  # Clear previous transactions viz

        # Set the current step to a random point within the data frame
        self.current_step=0
        self.current_date = random.randint(
            6, len(self.df.loc[:, 'Open'].values) - 1)
        self.init_date=self.current_date
        return self._next_observation()

    def render(self, mode='human', close=False):
        # visualization purposes
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_date}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')