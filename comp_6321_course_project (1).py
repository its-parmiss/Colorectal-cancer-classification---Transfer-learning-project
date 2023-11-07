# -*- coding: utf-8 -*-
"""Comp 6321 Course project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tDQ8W3Jw_oS-FzkoRKiHV30nCpFMtiZg

# imports
"""

import pickle
import torch
from google.colab import drive
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

"""# **Loading the data and pre-processing steps**"""

# Mount your Google Drive
drive.mount('/content/drive')

base_path = '/content/drive/My Drive/Comp6321 project dataset/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

datasets = {}

dataset_names = ['Colorectal Cancer', 'Prostate Cancer', 'Animal Faces']

for dataset_name in dataset_names:
    dataset_path = base_path + dataset_name
    dataset = ImageFolder(root=dataset_path, transform=transform)
    datasets[dataset_name] = dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



"""# **task 1: Colorectal Cancer Classification Using CNNs**

# Task 1 - part a : implementing ResNet-50 architecture and training from the scratch

## CNN Architecture
"""



"""## Model Training"""



"""## Model Optimization by hyperparameter grid search"""



"""## Results for Task 1 - Part a

### Evaluation
"""

def evaluate_classification_metrics(predictions, labels, loss,class_names):
    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    report = classification_report(labels, predictions, target_names=class_names)

    confusion = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Loss: {loss:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

"""### Visualization with T-SNE"""



"""# Task 1 - part b :using a pre-trained network for  Clorectal Cancer Classification

## loading the Pre-trained Network
"""

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

"""## Fine-tuning for Colorectal Cancer Classification"""

dataset = datasets['Colorectal Cancer']
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)

=
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Accuracy Curve')

plt.tight_layout()
plt.show()

model.fc = nn.Linear(2048, num_classes)  # 2048 is the number of features in the ResNet-50 classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

model.to(device)

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_classes = len(dataset.classes)
model.fc = nn.Linear(2048, num_classes)  # 2048 is the number of features in the ResNet-50 classifier


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# lists to store losses and accuracy for plotting the learning curve
train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_accuracy = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracy.append(epoch_train_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {epoch_train_accuracy}%")

model_filename = base_path + 'dataset1_model_using_pretrained_resnet50.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

"""## Results for Task 1 - part b

### Evaluation
"""

# Testing loop
model.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
predictions = []
labels = []

with torch.no_grad():
    for inputs, ground_truth in test_loader:
        inputs, ground_truth = inputs.to(device), ground_truth.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, ground_truth)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_test += ground_truth.size(0)
        correct_test += (predicted == ground_truth).sum().item()
        predictions.extend(predicted.cpu().numpy())
        labels.extend(ground_truth.cpu().numpy())

test_accuracy = 100 * correct_test / total_test
test_loss /= len(test_loader)

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

class_names = dataset.classes

evaluate_classification_metrics(predictions, labels, test_loss,class_names)

#todo : learning curve?

"""### T-SNE Visualization"""

model.eval()
all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        features = model(inputs).cpu().numpy()
        all_features.append(features)
        all_labels.append(labels.cpu().numpy())

all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

tsne = TSNE(n_components=2, random_state=0)
reduced_features = tsne.fit_transform(all_features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='viridis')
plt.title('t-SNE Visualization of Extracted Features')
plt.colorbar()
plt.show()



"""# **Task 2**

## using the CNN encoder trained in task 1

### extracting features
"""



"""###  t-SNE visualization"""



"""###  training ML models to classify the extracted features"""



"""#### k-nearest neighbors clustering (unsupervised learning)"""



"""#### Random Forrest(Supervised Learning)"""



"""## using pre-trained VGG16

### extracting features

note: we run the following cell once and store the features in drive, and after that we can load them from the drive.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = models.vgg16(pretrained=True).to(device)
feature_extractor = nn.Sequential(*list(pretrained_model.features.children())).to(device)  # Remove the last layer

# Define data transformations for normalization to ImageNet
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

d2_batch_size = 32
d3_batch_size = 32

prostate_data_loader = DataLoader(datasets['Prostate Cancer'], batch_size=d2_batch_size, shuffle=True)
animal_faces_data_loader = DataLoader(datasets['Animal Faces'], batch_size=d3_batch_size, shuffle=True)

# Initialize tqdm progress bars for feature extraction
prostate_pbar = tqdm(total=len(prostate_data_loader), desc="Prostate Features")
animal_faces_pbar = tqdm(total=len(animal_faces_data_loader), desc="Animal Faces Features")

# Extract features from the datasets
prostate_features = []
animal_faces_features = []

for data_loader, features_list, pbar in [(prostate_data_loader, prostate_features, prostate_pbar), (animal_faces_data_loader, animal_faces_features, animal_faces_pbar)]:
    feature_extractor.eval()  # Set the feature extractor to evaluation mode
    progress_counter = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            # Apply ImageNet normalization to the images before feature extraction
            images = normalize_transform(images)
            features = feature_extractor(images)
            features_list.append(features)
            progress_counter = progress_counter+1
            if progress_counter % 20 ==1:
              pbar.update(20)  # Update the progress bar
              progress_counter = 0

prostate_pbar.close()
animal_faces_pbar.close()

# Now train separate classifiers on top of the extracted features for each dataset

# shared_folder_path = base_path

# # Save the extracted features in the shared folder using pickle
# with open(shared_folder_path + 'prostate_features.pkl', 'wb') as file:
#     pickle.dump(prostate_features, file)

# with open(shared_folder_path + 'animal_faces_features.pkl', 'wb') as file:
#     pickle.dump(animal_faces_features, file)

shared_folder_path = base_path

with open(shared_folder_path + 'prostate_features.pkl', 'rb') as file:
    prostate_features = pickle.load(file)

with open(shared_folder_path + 'animal_faces_features.pkl', 'rb') as file:
    animal_faces_features = pickle.load(file)

"""###  t-SNE visualization"""

# Remove the last batch (incomplete batch) from the feature lists
prostate_features = prostate_features[:-1]
animal_faces_features = animal_faces_features[:-1]

# Convert the feature lists to NumPy arrays and flatten them
prostate_features = np.concatenate([feature.view(feature.size(0), -1).cpu().numpy() for feature in prostate_features], axis=0)
animal_faces_features = np.concatenate([feature.view(feature.size(0), -1).cpu().numpy() for feature in animal_faces_features], axis=0)

# Get the number of data points for each dataset after removing the last batch and only keep labels for the remaining data points
num_prostate_samples = prostate_features.shape[0]
num_animal_faces_samples = animal_faces_features.shape[0]

prostate_class_labels = [target for _, target in datasets['Prostate Cancer']][:num_prostate_samples]
animal_faces_class_labels = [target for _, target in datasets['Animal Faces']][:num_animal_faces_samples]

# T-SNE for prostate Cancer dataset
# model.eval()
# all_features = []
# all_labels = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         features = model(inputs).cpu().numpy()
#         all_features.append(features)
#         all_labels.append(labels.cpu().numpy())

# all_features = np.vstack(all_features)
# all_labels = np.concatenate(all_labels)

# tsne = TSNE(n_components=2, random_state=0)
# reduced_features = tsne.fit_transform(all_features)

# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='viridis')
# plt.title('t-SNE Visualization of Extracted Features')
# plt.colorbar()
# plt.show()

# tsne_prostate = TSNE(n_components=2, random_state=0)
# reduced_prostate_features = tsne_prostate.fit_transform(prostate_features)



# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_prostate_features[:, 0], reduced_prostate_features[:, 1], c=prostate_class_labels, cmap='viridis', label='Prostate Cancer')
# plt.title('t-SNE Visualization of Extracted Features (Prostate Cancer)')
# plt.legend()
# plt.colorbar()
# plt.show()

# # T-SNE for Animal Faces dataset
# tsne_animal_faces = TSNE(n_components=2, random_state=0)
# reduced_animal_faces_features = tsne_animal_faces.fit_transform(animal_faces_features)

# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_animal_faces_features[:, 0], reduced_animal_faces_features[:, 1], c=animal_faces_class_labels, cmap='viridis', label='Animal Faces')
# plt.title('t-SNE Visualization of Extracted Features (Animal Faces)')
# plt.legend()
# plt.colorbar()
# plt.show()

# T-SNE for prostate Cancer dataset
tsne_prostate = TSNE(n_components=2, random_state=0)
reduced_prostate_features = tsne_prostate.fit_transform(prostate_features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_prostate_features[:, 0], reduced_prostate_features[:, 1], c=prostate_class_labels, cmap='viridis', label='Prostate Cancer')
plt.title('t-SNE Visualization of Extracted Features (Prostate Cancer)')
plt.legend()
plt.colorbar()
plt.show()

# T-SNE for Animal Faces dataset
tsne_animal_faces = TSNE(n_components=2, random_state=0)
reduced_animal_faces_features = tsne_animal_faces.fit_transform(animal_faces_features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_animal_faces_features[:, 0], reduced_animal_faces_features[:, 1], c=animal_faces_class_labels, cmap='viridis', label='Animal Faces')
plt.title('t-SNE Visualization of Extracted Features (Animal Faces)')
plt.legend()
plt.colorbar()
plt.show()

"""###  training ML models to classify the extracted features"""



"""#### k-nearest neighbors clustering (unsupervised learning)"""



"""#### Random Forrest(Supervised Learning)"""

