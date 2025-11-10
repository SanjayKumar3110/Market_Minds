# core/Neural_NET.py
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        super(NeuralNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        # Define CNN layers for feature extraction
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        ).to(self.device)
        
        # Calculate the size of the output from the CNN layer
        conv_output_size = self.calculate_conv_output_size(sequence_length, input_dim)

        # Define LSTM layer for sequential modeling
        self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=hidden_dim, num_layers=1, batch_first=True).to(self.device)

        # Define the final linear layer
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.to(self.device)

        # Step 1: Permute dimensions for CNN
        x = x.permute(0, 2, 1)  # New shape: (batch_size, input_dim, sequence_length)

        # Step 2: Pass through CNN layer
        cnn_out = self.conv_layer(x)

        # Step 3: Reshape output for LSTM
        lstm_input = cnn_out.permute(0, 2, 1)  # New shape: (batch_size, new_seq_length, new_input_dim)

        # Step 4: Pass through LSTM layer
        lstm_out, _ = self.lstm(lstm_input)
        
        # Step 5: Pass the last time step's output to the final linear layer
        final_out = self.fc(lstm_out[:, -1, :])

        return final_out

    def train_step(self, state_tensor, target_tensor):
        self.optimizer.zero_grad()
        state_tensor = state_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        output = self.forward(state_tensor)
        loss = self.loss_fn(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_device(self):
        return self.device

    def calculate_conv_output_size(self, sequence_length, input_dim):
        # Create a dummy tensor to pass through the CNN layers
        dummy_input = torch.zeros(1, input_dim, sequence_length).to(self.device)
        conv_output = self.conv_layer(dummy_input)
        return conv_output.shape[1]