import torch
import random
import numpy as np

from collections import deque
from core.Neural_net import NeuralNetwork


class StrategyRLAgent:
    def __init__(self, sequence_length, input_dim, hidden_dim, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, buffer_size=10000, batch_size=32,
                 target_update=100):
        
        # Main model: The Q-network that is being trained
        self.model = NeuralNetwork(sequence_length, input_dim, hidden_dim, action_size, learning_rate)
        
        # Target model: A delayed copy of the main model for stable Q-learning
        self.target_model = NeuralNetwork(sequence_length, input_dim, hidden_dim, action_size, learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Set target network to evaluation mode

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        
        self.target_update_counter = 0
        self.target_update = target_update
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        # Reshape state to match the model's expected input shape: (1, sequence_length, input_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.model.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Unpack the batch into separate tensors for efficient processing
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Reshape states and next_states to be a single batch tensor (batch_size, sequence_length, input_dim)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.model.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.model.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.model.device)

        # Get Q-values for the current states from the main model
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max Q-values for the next states from the TARGET model
        # Use target_model for stability
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            # Set next_q_values to 0 for terminal states (where done is True)
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Calculate loss and perform backpropagation
        loss = self.model.loss_fn(current_q_values, target_q_values)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.model.device))
        self.model.eval()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
