# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="787" height="787" alt="image" src="https://github.com/user-attachments/assets/13a1ab90-3062-45c3-8ed3-0ac72833deae" />


## DESIGN STEPS

### STEP 1:
Import necessary libraries and load the dataset.

### STEP 2:
Encode categorical variables and normalize numerical features.

### STEP 3:
Split the dataset into training and testing subsets.
### STEP 4:
Design a multi-layer neural network with appropriate activation functions.
### STEP 5:
Train the model using an optimizer and loss function.
### STEP 6:
Evaluate the model and generate a confusion matrix.
### STEP 7:
Use the trained model to classify new data samples.
### STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM
### Name: Niralya J
### Register Number:212224230188


```
class PeopleClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
    
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information
<img width="1492" height="290" alt="image" src="https://github.com/user-attachments/assets/94fe2b85-add9-4ec2-b36d-52c5a1833254" />



## OUTPUT
### Confusion Matrix
<img width="772" height="654" alt="image" src="https://github.com/user-attachments/assets/c59ed3b2-9df2-4a61-85a6-78e243f9037a" />


### Classification Report
<img width="812" height="442" alt="image" src="https://github.com/user-attachments/assets/17040b53-d025-48cb-822c-050535782301" />




### New Sample Data Prediction
<img width="1197" height="323" alt="image" src="https://github.com/user-attachments/assets/02b336ec-3edb-42c9-98ee-440d92bbf078" />



### Result
Thus the neural network classification model was successfully developed.
