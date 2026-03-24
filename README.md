# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Image classification is the task of assigning a label to an input image from a predefined set of categories.

In this experiment, a CNN model is developed using PyTorch to classify grayscale fashion product images into 10 categories using the Fashion-MNIST dataset.

## Dataset Details:

Dataset Name: Fashion-MNIST

Total Training Images: 60,000

Total Test Images: 10,000

Image Size: 28 × 28 pixels

Number of Classes: 10

## Classes:

T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot

## Neural Network Model

<img width="1347" height="490" alt="image" src="https://github.com/user-attachments/assets/1c3e3ef2-9665-42f5-b8ef-f55f6e4254e8" />


## DESIGN STEPS
## STEP 1:

Load and preprocess the Fashion-MNIST dataset. Apply transformations such as tensor conversion and normalization.

## STEP 2:

Design the CNN architecture using convolutional, pooling, activation, and fully connected layers.

## STEP 3:

Train the model using CrossEntropy loss and Adam optimizer. Evaluate the model using accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: Kevin P
### Register Number: 212224040159
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x



```

```python
# Initialize the Model, Loss Function, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: Kevin P')
        print('Register Number: 212224040159')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

<img width="508" height="279" alt="image" src="https://github.com/user-attachments/assets/1b866e55-0941-46b2-82a7-361923f8afc6" />



### Confusion Matrix

<<img width="536" height="582" alt="image" src="https://github.com/user-attachments/assets/529c96ab-96ed-4b6a-8866-eddc647c601d" />


### Classification Report

<img width="675" height="448" alt="image" src="https://github.com/user-attachments/assets/1847df8b-03b5-4ceb-9496-97b22990ba73" />
/>



### New Sample Data Prediction

<img width="638" height="722" alt="image" src="https://github.com/user-attachments/assets/ac32c489-19d4-4165-b8c6-c0d3ebf3f13c" />

<img width="766" height="711" alt="image" src="https://github.com/user-attachments/assets/e9c6a517-0a8b-4262-9d17-9214719e9576" />



## RESULT
Thus, a Convolutional Deep Neural Network was successfully implemented using the Fashion-MNIST dataset. The model achieved good classification accuracy and was able to correctly classify new sample images.
