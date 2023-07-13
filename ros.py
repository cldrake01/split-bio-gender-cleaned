import os
from time import sleep

import cv2
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


# @lru_cache(maxsize=None)
# def resample() -> tuple[np.ndarray, np.ndarray]:
#     return ros.fit_resample(train_data, train_data.targets)


data_dir: str = '/Users/collin/Downloads/pre-ros-split-bio-gender-cleaned'

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
ros = RandomOverSampler(
    sampling_strategy='not majority'
)

print("Original training data size: ", len(train_data.targets))

train_data_flat: np.ndarray = np.array(train_data.imgs, dtype=object).reshape(-1, 1)
test_data_flat: np.ndarray = np.array(test_data.imgs, dtype=object).reshape(-1, 1)

train_data_resampled, train_labels_resampled = ros.fit_resample(train_data, train_data.targets)

print("Resampled training data size: ", len(train_data_resampled))

# Set the directory to save the new images
save_dir = 'post-ros-split-bio-gender-cleaned/train'
os.makedirs(save_dir, exist_ok=True)

print("len: ", len(train_data_resampled))
print("range: ", range(len(train_data_resampled)))

print("print(train_data_resampled)")
print(train_data_resampled)
print("print(train_data_resampled[0])")
print(train_data_resampled[0])
print("print(train_data_resampled[1])")
print(train_data_resampled[1])
print("print(train_data_resampled[0][0])")
print(train_data_resampled[0][0])
print("print(train_data_resampled[1][0])")
print(train_data_resampled[1][0])
print("print(train_data_resampled[0][-1])")
print(train_data_resampled[0][-1])
print("print(train_data_resampled[1][-1])")
print(train_data_resampled[1][-1])
print("print(train_data_resampled[-1][0])")
print(train_data_resampled[-1][0])
print("print(train_data_resampled[-1][-1])")
print(train_data_resampled[-1][-1])

for f in torch.unsqueeze(train_data_resampled[0][0], 0):

    print("f: ", f)

    for i in range(len(train_data_resampled)):
        # Get image and label
        image: torch.Tensor = train_data_resampled[i][0]
        label: int = train_data_resampled[i][1]

        # Ensure the image tensor has 3 channels (RGB)
        if image.shape[0] != 3:
            image = image.repeat(3, 1, 1)

        # Convert the tensor to a numpy array
        image_np = image.permute(1, 2, 0).numpy()  # Assumes the tensor shape is (channels, height, width)

        # Convert the numpy array to BGR format (required by OpenCV)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        image_bgr = cv2.equalizeHist(image_bgr)

        # Extract the relevant directory information
        gender_age_dir = train_data_resampled[i][0][0]
        category_dir = train_data.classes + gender_age_dir

        # Create the new directories if they don't exist
        save_category_dir = os.path.join(save_dir, category_dir)
        os.makedirs(save_category_dir, exist_ok=True)

        filename: str = r'{}-{}-{}.jpg'.format(category_dir, label, i)

        print("filename: ", filename)

        # Create the new path for saving
        new_path: str = os.path.join(save_category_dir, filename)

        # Save image to new path using OpenCV
        cv2.imwrite(new_path, image_bgr)

        print("Saving image: ", new_path)

        if i % 23323.0 == 0.0:
            f += 1
            print("f: ", f)

print("Done saving images to new directory")

# for i in range(len(train_data_resampled)):
#     # Get image and label
#     image: torch.Tensor = train_data_resampled[i][0]
#     label: int = train_data_resampled[i][1]
#
#     # Ensure the image tensor has 3 channels (RGB)
#     if image.shape[0] != 3:
#         image = image.repeat(3, 1, 1)
#
#     # Get image path
#     path: str = train_data_resampled[i][0]
#
#     # Get image filename
#     filename: str = os.path.basename(path)
#
#     # Extract the relevant directory information
#     gender_age_dir = os.path.dirname(path)
#     category_dir = os.path.basename(gender_age_dir)
#
#     # Create the new directories if they don't exist
#     save_dir = os.path.join('post-ros-split-bio-gender-cleaned/train', category_dir)
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Create the new path for saving
#     new_path: str = os.path.join(save_dir, filename)
#
#     image = image * 80
#
#     # Convert the tensor to a numpy array
#     image_np = image.permute(1, 2, 0).numpy()  # Assumes the tensor shape is (channels, height, width)
#
#     # Convert the numpy array to BGR format (required by OpenCV)
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#
#     # Save image to new path using OpenCV
#     cv2.imwrite(new_path, image_bgr)
#
#     # print("Saving image: ", new_path)
#
# print("Done saving images to new directory")
