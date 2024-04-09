# Import Statements
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters and Device Setup
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size): # Initialize the replay buffer with specified sizes.
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done): # Add an experience to the replay buffer.
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self): # Sample a batch of experiences from the replay buffer and format them into PyTorch tensors.
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self): # Return the current size of the internal memory.
        return len(self.memory)


class Agent:
    def __init__(self, state_size, action_size):
        """
        Initialize the agent with state and action sizes, 
        two Q-Networks (local and target), 
        an optimizer for the local network, 
        a replay buffer, a step counter, and an inventory.
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay Buffer
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE)
        # Initialize step
        self.t_step = 0
        # Initialize inventory
        self.inventory = []

    def step(self, state, action, reward, next_state, done):
        # Store the experience in replay memory.
        self.memory.add(state, action, reward, next_state, done)

        # self.t_step will count from 0 to UPDATE_EVERY - 1, and then reset to 0, repeatedly.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def learn(self, experience, gamma):
        """
        Process a batch of experiences to update the local Q-Network.
        Then soft update the target Q-Network parameters towards the local Q-Network parameters.
        """
        states, actions, rewards, next_states, dones = experience

        # target network
            # compute the maximum Q-value for the next states from the target network.
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # calculates the target Q-values for each experience in the batch. 
            # use the Bellman equation: 
            # the immediate reward plus the discounted maximum future Q-value, adjusting for terminal states with (1 - dones). 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # compute the Q-values predicted by the local network for the action taken in each state.
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())

        # The loss is calculated as the mean squared error (MSE) between Q_expected and Q_targets, 
            # representing the difference between predicted Q-values and the target Q-values. 
            # This loss reflects how well the local Q-network is predicting the Q-values that would maximize future rewards.
        loss = F.mse_loss(Q_expected, Q_targets)

        # Before backpropagation, self.optimizer.zero_grad() clears any gradients accumulated from previous 
            # operations to ensure the current update is only based on the latest computed loss.
        self.optimizer.zero_grad()
        # compute the gradient of the loss with respect to all trainable parameters in the local Q-network.
        loss.backward()
        # update the weights of the local Q-network using the gradients calculated by loss.backward().
        self.optimizer.step()

        # updates the weights of the target network to slowly track the local network. 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, state, eps = 0.):
        """select an action for the agent to take, based on the current state of the environment and the agent's policy. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))