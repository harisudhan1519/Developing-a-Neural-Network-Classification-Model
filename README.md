## Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="882" height="958" alt="image" src="https://github.com/user-attachments/assets/52f3ada5-70ce-41ab-83cb-62184079cd89" /> 


## DESIGN STEPS
## STEP 1:
Load the customer dataset and preprocess it by handling missing values and encoding categorical features.

## STEP 2:
Split the dataset into training and testing sets to evaluate model performance.

## STEP 3:
Define a neural network architecture with fully connected layers and ReLU activation functions.

## STEP 4:
Select an appropriate loss function (CrossEntropyLoss) and optimizer (Adam) for multi-class classification.

## STEP 5:
Train the neural network using the training data through forward pass, loss computation, and backpropagation.

## STEP 6:
est the trained model on unseen data and predict the customer segment (A, B, C, or D).

## PROGRAM

### Name: Mohammed Hari sudhan S

### Register number 212224240048

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32 , 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1293" height="259" alt="image" src="https://github.com/user-attachments/assets/7b72e6c9-24f4-45c8-83d1-3d2162566ff4" />

### OUTPUT

## Confusion Matrix

<img width="671" height="573" alt="image" src="https://github.com/user-attachments/assets/730a8e8b-5be3-4ffc-b455-cd8883f30041" />

## Classification Report
<img width="1523" height="1032" alt="ChatGPT Image May 11, 2026, 01_28_26 PM" src="https://github.com/user-attachments/assets/2634cda1-ff22-4f22-b48a-043916dc0ff0" />

### New Sample Data Prediction
<img width="1576" height="422" alt="image" src="https://github.com/user-attachments/assets/fdd47f2f-1b6d-46a8-bc26-4b58dce507a8" />

## RESULT
<img width="1021" height="65" alt="image" src="https://github.com/user-attachments/assets/6d9dc626-8130-40b6-9657-2211f6bf3e94" />

