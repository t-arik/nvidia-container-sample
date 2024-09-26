import torch
import torch.nn as nn
import torch.optim as optim
import time  # Import the time module

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
scale = 8

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(scale*1024, scale*2048)
        self.fc2 = nn.Linear(scale*2048, scale*2048)
        self.fc3 = nn.Linear(scale*2048, scale*1024)
        self.fc4 = nn.Linear(scale*1024, scale*10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model, move it to GPU, and define loss and optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate random data
batch_size = 256
input_data = torch.randn(batch_size, scale*1024).to(device)
target_data = torch.randint(0, 10, (batch_size,)).to(device)

# Training loop
num_epochs = 10000
start_time = time.time()  # Start time
interval = 2  # Time interval in seconds

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero the gradient buffers
    output = model(input_data)  # Forward pass
    loss = criterion(output, target_data)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    # Check if 2 seconds have passed
    if time.time() - start_time >= interval:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        start_time = time.time()  # Reset the timer

print('Training complete')
