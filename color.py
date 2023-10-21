import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

model = SimpleNet()

# Simple training data
torch.manual_seed(42)

# Generate 100 random x and y coordinates
train_data = torch.rand(100, 3)
# Generate 100 random RGB values

train_labels = torch.rand(100, 3)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = loss_fn(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(loss)


import pygame
import torch

# Pygame initialization
pygame.init()
win_width, win_height = 400, 400
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Neural Network Color Visualization")

def predict_color(x, y, model):
    with torch.no_grad():
        input_vector = torch.zeros(1, 3)
        input_vector[0, 0] = x / 400.0
        input_vector[0, 1] = y / 400.0
        output = model(input_vector)
        return (output.numpy()[0] * 255).astype(int)  # Convert color range from [0,1] to [0,255]

# Fill the window with colors based on model predictions
x_coords = torch.linspace(0, 1, 400).view(-1, 1).repeat(1, 400).view(-1)
y_coords = torch.linspace(0, 1, 400).repeat(400)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    time_val = np.sin(time.time() * 0.2)

    # Integrate time value into the grid
    grid_time_value = torch.tensor([time_val] * 400 * 400, dtype=torch.float32)  # Explicitly set dtype
    grid = torch.stack([x_coords, y_coords, grid_time_value], dim=1)

    # Predict colors for the entire grid

    with torch.no_grad():
        colors = model(grid).numpy() * 255
    colors = colors.reshape(400, 400, 3).astype(int)

    for x in range(400):
        for y in range(400):
            win.set_at((x, y), tuple(colors[y, x]))

    pygame.display.flip()

pygame.quit()

