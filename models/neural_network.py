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
df = pd.read_csv('data/diabetes_indicator.csv')

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
        self.fc1 = nn.Linear(19, 16)
        # Second layer with 16 inputs and 8 outputs
        self.fc2 = nn.Linear(16, 8)
        # Output layer with 3 output classes (no diabetes, pre-diabetes, diabetes)
        self.fc3 = nn.Linear(8, 3)

    # Our forward pass function utilizing RELU
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the train_model function
def train_model(model, train_loader, epochs=20):
    # Cross entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer for updating weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'diabetes_model.pth')
    print("Model saved!")
    return model


# Define the evaluate_model function
def evaluate_model(model, X_test_tensor, y_test):
    """
    Evaluates the trained model on the test dataset.
    Args:
        model: Trained PyTorch model.
        X_test_tensor: Test data features as PyTorch tensor.
        y_test: True labels for the test data.

    Returns:
        accuracy: Model accuracy on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test_tensor)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get class predictions
        accuracy = accuracy_score(y_test, predicted.numpy()) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Main training and evaluation logic
if __name__ == "__main__":
    model = DiabetesModel()  # Initialize the model

    # Train the model
    print("Starting training...")
    trained_model = train_model(model, train_loader)

    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(trained_model, X_test_tensor, y_test)
