import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler


def random_over_sampling(data_dir: str, save_dir: str = 'post-ros-split-bio-gender-cleaned/train') -> None:
    """
    Resample the training data to handle class imbalance issue. The resampled data will be saved to the specified
    directory. Note: This function is only for the training data. The validation and test data should not be resampled.
    :param data_dir: The directory of the training data, e.g.
     '/Users/collindrake/Downloads/pre-ros-split-bio-gender-cleaned'
    :param save_dir: The directory to which the final results will be saved. The default parameter is sufficient for
     Linux and macOS users.
    :return: None
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(root=data_dir + '/train', transform=transform)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

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

    os.makedirs(save_dir, exist_ok=True)

    file: int = 0

    for folder in tqdm(X, desc="Resampling", total=len(X)):

        file += 1

        str_file: str = f'{file}'

        # os.makedirs(os.path.join(save_dir, folder))

        for image_index in range(len(folder)):

            image, label = folder[image_index]

            str_label: str = f'{label}'

            os.makedirs(save_dir + '/' + str_label, exist_ok=True)

            image *= 255

            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)

            image_bgr = image.permute(1, 2, 0).numpy()
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # image_bgr = cv2.equalizeHist(image_bgr)

            filename: str = f'{str_file}.jpg'

            final_path = os.path.join(save_dir, str_label, filename)

            cv2.imwrite(final_path, image_rgb)

            # print("Saving image:", final_path)

    print("Resampling: Complete")


random_over_sampling('/Users/collindrake/Downloads/pre-ros-split-bio-gender-cleaned')
