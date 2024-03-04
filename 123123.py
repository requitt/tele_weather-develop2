import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Let's write a simple GAN class
class TestImage(nn.Module):
    def __init__(self):
        super(TestImage, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the generator
test_gen = TestImage()

# Generate a random image
noise = torch.randn(1, 100)
random_image = test_gen(noise).detach().numpy().reshape(1, 3, 1)

# Display the image
plt.imshow(random_image)
plt.axis('off')
plt.show()