import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load dataset
df = pd.read_csv('diabetes_indicator.csv')

# Prepare features and target
X = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values  # Replace with your feature columns
y = df['Diabetes_012'].values 

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        # First layer with 19 input features and 16 neurons
        self.fc1 = nn.Linear(19,16)
        # Second layer with 16 inputs and 8 outputs
        self.fc2 = nn.Linear(16,8)
        # Output player with 3 output classes (no diabetes, pre-diabetes, diabetes)
        self.fc3 = nn.Linear(8,3)
    # Our forward pass function utilizing RELU
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = DiabetesModel()
# Cross entropy loss (our loss function) is great for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam is optimizer that adjusts learning rate during training
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# We are done training and ready to evaluate
model.eval()
with torch.no_grad():
    # Forward pass to get the predictions (logits) from the model. The model outputs a tensor with raw prediction values for each class (0, 1, 2).
    outputs = model(X_test_tensor)
    # Outputs is a row (tensor row) that contains certainty values for each class (0, 1, 2). We pick the highest certainty (_ is for highest value as we don't needm predicted is the index)
    _, predicted = torch.max(outputs, 1)
    # y_test contains the actual results (diabetes) and compares it with our predicted
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save our model for actual implementation
torch.save(model.state_dict(), 'diabetes_model.pth')
print("Model saved!")
