import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.utils.data
import torch.optim as optim
from functools import lru_cache
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler


class GenderRecognition(nn.Module):
    def __init__(self, num_classes_inner: int):
        super(GenderRecognition, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


@lru_cache(maxsize=None)
def resample() -> tuple[np.ndarray, np.ndarray]:
    return ros.fit_resample(train_data, train_data.targets)


data_dir: str = '/Users/collin/Downloads/split-bio-gender-cleaned'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data: ImageFolder = ImageFolder(root=data_dir + '/train', transform=transform)
test_data: ImageFolder = ImageFolder(root=data_dir + '/test', transform=transform)
val_data: ImageFolder = ImageFolder(root=data_dir + '/val', transform=transform)

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size: int = 32
train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Apply Random Oversampling (ROS) to the training data
ros = RandomOverSampler()

print("Original training data size: ", len(train_data.targets))

train_data_flat: np.ndarray = np.array(train_data.imgs, dtype=object).reshape(-1, 1)
test_data_flat: np.ndarray = np.array(test_data.imgs, dtype=object).reshape(-1, 1)

train_data_resampled, train_labels_resampled = resample()

print(train_data_resampled)

os.makedirs('ros-split-bio-gender-cleaned' + '/train', exist_ok=True)
os.makedirs('ros-split-bio-gender-cleaned' + '/test', exist_ok=True)

print("Resampled training data size: ", len(train_data_resampled))

# Write resampled training images to new directory
# Write resampled training images to new directory
for i in range(len(train_data_resampled)):
    # Get image and label
    image: np.ndarray = train_data_resampled[i][0]
    label: int = train_data_resampled[i][1]
    # Get image path
    path: str = train_data.imgs[i][0]
    # Get image filename
    filename: str = os.path.basename(path)
    # Get image folder
    folder: str = os.path.dirname(path)
    # Get image extension
    extension: str = os.path.splitext(filename)[1]
    # Create new path
    new_path: str = folder + '/' + filename

    # Convert the tensor to a NumPy array
    image_array = image.numpy()

    # Convert the NumPy array to a PIL image
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

    # Save image to new path
    image_pil.save(new_path)

    print("Saving image: ", new_path)
