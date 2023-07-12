import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler

data_dir = '/Users/collin/Downloads/split-bio-gender-cleaned'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root=data_dir + '/train', transform=transform)
test_data = ImageFolder(root=data_dir + '/test', transform=transform)
val_data = ImageFolder(root=data_dir + '/val', transform=transform)

# Apply Random Oversampling (ROS) to the training data
# Exclude 'female-20-29' and 'male-30-49' classes
ros = RandomOverSampler(
    sampling_strategy={'female-0-12': 19_000, 'female-13-19': 19_000, 'female-30-49': 19_000, 'female-50-64': 19_000,
                       'female-65-100': 19_000, 'male-0-12': 19_000, 'male-13-19': 19_000, 'male-20-29': 19_000,
                       'male-50-64': 19_000, 'male-65-100': 19_000}
)
train_data_resampled, train_labels_resampled = ros.fit_resample(train_data, train_data.targets)

print("Original training data size: ", len(train_data.targets))

os.makedirs('split-bio-gendered-cleaned-ros', exist_ok=True)
os.makedirs('split-bio-gendered-cleaned-ros/train', exist_ok=True)
os.makedirs('split-bio-gendered-cleaned-ros/test', exist_ok=True)

# For each directory in the training data, create a new directory in the resampled training data
# and save the images to the new directory
for directory in os.listdir(data_dir + '/train'):
    print(directory)
    os.makedirs('split-bio-gendered-cleaned-ros/train/' + directory, exist_ok=True)
    for file in os.listdir(data_dir + '/train/' + directory):
        os.rename(data_dir + '/train/' + directory + '/' + file,
                  'split-bio-gendered-cleaned-ros/train/' + directory + '/' + file)

# Then write the files to the new directories
for i in range(len(train_data_resampled)):
    print(i)
    image, label = train_data_resampled[i]
    image.save('split-bio-gendered-cleaned-ros/train/' + train_data.classes[label] + '/' + str(i) + '.jpg')
