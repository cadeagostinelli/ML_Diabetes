import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv('../data/diabetes_indicator.csv')

# Prepare features and target
feature_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
                   'DiffWalk', 'Sex', 'Age']

X = df[feature_columns].values  
y = df['Diabetes_012'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Now X_train and X_test have the same columns as feature_columns

print(f"X shape before split: {X.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"X_train_tensor shape: {X_train_tensor.shape}")
# After converting to PyTorch tensors
print("Class distribution in training set:", y_train_tensor.bincount())
print("Class distribution in test set:", y_test_tensor.bincount())

# Calculate class weights based on the inverse frequency of classes
class_counts = torch.tensor([170908, 3687, 28349])
class_weights = 1.0 / class_counts
normalized_weights = class_weights / class_weights.sum()  # Normalize weights
print(f"Normalized class weights: {normalized_weights}")

# Apply these weights to the CrossEntropyLoss function
criterion = nn.CrossEntropyLoss(weight=normalized_weights)

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


def train_model(model, train_loader, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'diabetes_model.pth')
    print("Model saved!")
    return model


def evaluate_model(model, X_test_tensor, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.numpy()) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Debugging added for prediction consistency
def debug_prediction_input(input_df, expected_features):
    print(f"Input DataFrame columns: {input_df.columns.tolist()}")
    missing = [feature for feature in expected_features if feature not in input_df.columns]
    unexpected = [col for col in input_df.columns if col not in expected_features]
    if missing:
        raise ValueError(f"Missing features during prediction: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected features during prediction: {unexpected}")
    print("Prediction input matches training features.")


# Main training and evaluation logic
if __name__ == "__main__":
    model = DiabetesModel()

    # Train the model
    print("Starting training...")
    trained_model = train_model(model, train_loader)

    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(trained_model, X_test_tensor, y_test)
