import os
import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler

data_dir: str = '/Users/collindrake/Downloads/pre-ros-split-bio-gender-cleaned'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = ImageFolder(root=data_dir + '/train', transform=transform)
test_data = ImageFolder(root=data_dir + '/test', transform=transform)
val_data = ImageFolder(root=data_dir + '/val', transform=transform)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Handle class imbalance issue by using weighted loss function or other techniques

ros = RandomOverSampler(sampling_strategy='not majority')
train_loader_dataset = np.fromiter(train_loader.dataset, dtype=object)
train_loader_dataset = np.reshape(train_loader_dataset, (-1, 1))
train_data.targets = np.array(train_data.targets)
train_data.targets = np.reshape(train_data.targets, (-1, 1))

print(train_loader_dataset.shape)
print(train_data.targets.shape)

print("Original training data size: ", len(train_data.targets))

X, y = ros.fit_resample(train_loader_dataset, train_data.targets)

print("Resampled training data size: ", len(X))

save_dir = 'post-ros-split-bio-gender-cleaned/train'
os.makedirs(save_dir, exist_ok=True)

print("len: ", len(X))

gender_age_dir = 0

for folder in X:

    gender_age_dir = gender_age_dir + 1

    str_gender_age_dir = f'{gender_age_dir}'

    # os.makedirs(os.path.join(save_dir, folder))

    file_number = 0

    for image_index in range(len(folder)):
        file_number = file_number + 1

        image, label = folder[image_index]

        if image.shape[0] != 3:
            image = image.repeat(3, 1, 1)

        image_bgr = image.permute(1, 2, 0).numpy()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_bgr = cv2.equalizeHist(image_bgr)

        str_file_number = f'{file_number}'

        filename = r'{}-{}-{}.jpg'.format(label, image_index, str_file_number)

        final_path = os.path.join(save_dir, str_gender_age_dir, filename)

        cv2.imwrite(final_path, image_rgb)
        print("Saving image: ", final_path)

print("Done saving images to new directory")
