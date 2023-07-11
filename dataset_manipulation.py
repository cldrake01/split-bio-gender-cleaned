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
train_data_resampled, train_labels_resampled = ros.fit_resample(train_data.data, train_data.targets)

file_path = r'/Users/collin/Downloads/ros-split-bio-gender-cleaned'

for file in train_data_resampled:
    os.makedirs(file_path, exist_ok=True)
    with open(os.open(file_path, os.O_CREAT | os.O_WRONLY, 0o777), 'w') as fh:
        fh.write(file)

# The default umask is 0o22 which turns off write permission of group and others
# os.umask(0)
# with open(os.open(file_path, os.O_CREAT | os.O_WRONLY, 0o777), 'w') as fh:
#     fh.write('content')

# train_data_resampled = torch.from_numpy(train_data_resampled)
# train_labels_resampled = torch.from_numpy(train_labels_resampled)
#
# train_dataset_resampled = torch.utils.data.TensorDataset(train_data_resampled,train_labels_resampled)
#
# train_loader_resampled = torch.utils.data.DatraLoader(train_dataset_resampled, batch_size=batch_size, shuffle=True)
#
# num_classes = len(train_data.classes)
# model = GenderRecognition(num_classes).to(device)
#
# criterion = nn.CrossEntropyLoss().to(device)
#
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
