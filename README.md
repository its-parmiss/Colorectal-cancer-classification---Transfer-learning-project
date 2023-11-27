# Colorectal Cancer Detection Code Explanation

This code is designed to create a model for detecting colorectal cancer using PyTorch. 

## Library Import
The code begins by importing necessary libraries such as `torch`, `numpy`, `torchvision`, `seaborn`, and `sklearn`. These libraries provide functionalities for array operations, deep learning, data visualization, and machine learning respectively.

## Mounting Google Drive
The `drive.mount('/content/drive')` command is used to mount Google Drive to the Colab notebook. This allows the notebook to access files stored in your Google Drive.
To Do: this part of code needs to be updated to import dataset from a link with no need to drive mounting

## Dataset Path
The `colorectral_dataset_path` variable is set to the path where the colorectal cancer dataset is stored in your Google Drive.

## Creating Torchvision ImageFolder Dataset
The `ImageFolder` function from the `torchvision.datasets` module is used to load the dataset from the specified path. The images are resized to 224x224 pixels and converted to tensors using the `transforms` module.

## Train, Test, Validation Splitting
The dataset is split into training, validation, and testing sets in a 70:15:15 ratio using the `random_split` function from the `torch.utils.data` module. Dataloaders for each set are created using the `DataLoader` function, which will load the data in batches of 16 images and shuffle them.

## Defining the Colorectal Cancer Detection Class
The `ImageClassifier` class is defined to handle the creation and training of the model. The class includes methods for:
- Creating a ResNet model with a fully connected layer adjusted to the number of classes in the dataset.
- Defining the loss function as CrossEntropyLoss.
- Defining the optimizer as Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9.
- Defining the learning rate scheduler, which decays the learning rate by a factor of 0.1 every 7 epochs.
- Training the model for a specified number of epochs, keeping track of the best model weights based on validation accuracy.

The training process involves iterating over the training and validation data, calculating the loss and gradients, updating the model parameters, and tracking the training and validation loss and accuracy. The model with the highest validation accuracy is saved and returned after training is complete.

## Plotting Loss and Accuracy
The final part of the code is dedicated to visualizing the training and validation loss and accuracy. This is done using the `matplotlib.pyplot` module, which provides functions for creating figures and plotting data.

Two subplots are created in a single figure. The first subplot displays the training and validation loss over the epochs. The second subplot displays the training and validation accuracy over the epochs. Both plots include a legend to distinguish between the training and validation data.

The `plt.show()` function is called at the end to display the figure. This visualization can help in understanding how well the model is learning and generalizing to unseen data over time.

## Saving, Loading, and Evaluating the Model
The next part of the code involves saving the best model, loading it for evaluation, and calculating performance metrics.

### Saving the Best Model
The best model's weights are saved to a file using `torch.save()`. The path for the saved model is defined by the `best_model_path` variable.

### Loading Model and Evaluating on Test Set
The `evaluate_model_on_test_set()` function is defined to load the saved model and evaluate it on the test set. The function loads the model, sets it to evaluation mode, and then iterates over the test data. For each batch of images, it performs a forward pass through the model, gets the model's predictions, and stores them along with the true labels.

### Scores and Confusion Matrix
The model's predictions on the test set are used to calculate the F1 score, precision, and recall using functions from the `sklearn.metrics` module. These metrics provide a quantitative measure of the model's performance.

A confusion matrix is also computed and displayed using `seaborn`'s heatmap function. The confusion matrix provides a visual representation of the model's performance, showing the number of correct and incorrect predictions for each class.



