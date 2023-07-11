import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from imblearn.over_sampling import RandomOverSampler


class GenderRecognition(nn.Module):
    def __init__(self, num_classes_inner):
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

    def forward(self, x):
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


data_dir = '/Users/collin/Downloads/split-bio-gender-cleaned'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root=data_dir + '/train', transform=transform)
test_data = ImageFolder(root=data_dir + '/test', transform=transform)
val_data = ImageFolder(root=data_dir + '/val', transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Apply Random Oversampling (ROS) to the training data
# Exclude 'female-20-29' and 'male-30-49' classes
ros = RandomOverSampler(
    sampling_strategy={'female-0-12': 19_000, 'female-13-19': 19_000, 'female-30-49': 19_000, 'female-50-64': 19_000,
                       'female-65-100': 19_000, 'male-0-12': 19_000, 'male-13-19': 19_000, 'male-20-29': 19_000,
                       'male-50-64': 19_000, 'male-65-100': 19_000}
)
train_data_resampled, train_labels_resampled = ros.fit_resample(train_data.data, train_data.targets)

train_data_resampled = torch.from_numpy(train_data_resampled)
train_labels_resampled = torch.from_numpy(train_labels_resampled)

train_dataset_resampled = torch.utils.data.TensorDataset(train_data_resampled,train_labels_resampled)

train_loader_resampled = torch.utils.data.DatraLoader(train_dataset_resampled, batch_size=batch_size, shuffle=True)

num_classes = len(train_data.classes)
model = GenderRecognition(num_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader_resampled:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute the accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, val_loss, val_accuracy))

# Test loop
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# Compute the test accuracy
test_accuracy = 100 * test_correct / test_total

print('Test Accuracy: {:.2f}%'.format(test_accuracy))
