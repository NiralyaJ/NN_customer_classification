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
<img width="1227" height="357" alt="image" src="https://github.com/user-attachments/assets/a9ed1b37-c2fc-4b3a-9f5f-0b60a14763a7" />




## OUTPUT
### Confusion Matrix
<img width="775" height="763" alt="image" src="https://github.com/user-attachments/assets/3a62aceb-47a6-4132-bafb-7108056c34e5" />


### Classification Report
<img width="735" height="427" alt="image" src="https://github.com/user-attachments/assets/b1e49099-aff9-42e0-afd8-5fe60106edd6" />






### New Sample Data Prediction
<img width="487" height="217" alt="image" src="https://github.com/user-attachments/assets/f9b2bfae-39ec-4b18-84a8-149ca618e5bb" />




### Result
Thus the neural network classification model was successfully developed.
