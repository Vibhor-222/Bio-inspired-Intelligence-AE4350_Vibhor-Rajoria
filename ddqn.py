import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections # For dequeue for the memory buffer
import random
import dill as pickle # For storing the buffer state

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Replay memory to store and retrieve agent's experiences
class ReplayMemory(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter = 0  # Number of transitions in the memory
        self.index = 0  # Current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    # Save a transition into the memory
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    # Randomly sample transitions from the memory
    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size  # Should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size)  # Number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

# Neural network model for the Q-function approximation
class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# Base Agent class with common methods and attributes
class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000):
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_dec = epsilon_dec  # Exploration rate decrement for larger spaces
        self.epsilon_min = epsilon_end  # Minimum exploration rate
        self.batch_size = batch_size  # Batch size for learning from experiences
        self.memory = ReplayMemory(mem_size)  # Replay memory for storing agent's experiences

    # Save a transition (state, action, reward, new_state, done) into the memory
    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)

    # Choose an action based on the current state
    def choose_action(self, state):
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_func.eval()  # Set the network in evaluation mode
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()  # Set the network back in training mode
        if rand > self.epsilon:  # Exploration vs exploitation based on epsilon
            return np.argmax(action_values.cpu().data.numpy())  # Choose action with max Q-value
        else:
            return np.random.choice([i for i in range(4)])  # Explore by choosing a random action

    # Reduce the exploration rate over time
    def reduce_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        raise Exception("Not implemented")  # Placeholder for learning method, to be implemented by child classes

        
        
# Deep Q-Network (DQN) Agent
class DQN(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(8, 4, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size:  # Wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values using the Bellman equation
        q_next = self.q_func(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main neural network using mean squared error loss
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Reduce the exploration rate
        self.reduce_epsilon()


# Double Deep Q-Network (DDQN) Agent
class DDQN(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(8, 4, 42).to(device)
        self.q_func_target = QNN(8, 4, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size:  # Wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values using the Bellman equation
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main neural network using mean squared error loss
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Update the target neural network (every N-th step)
        if self.memory.trans_counter % self.replace_q_target == 0:
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
                
        # 5. Reduce the exploration rate
        self.reduce_epsilon()


