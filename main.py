import torch
from models.neural_network import DiabetesModel, train_model, evaluate_model
from utils.preprocess import prepare_data
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train_and_evaluate():
    """
    Function to train the model and evaluate its performance.
    Returns:
        trained_model: The trained PyTorch model.
        accuracy: Accuracy of the model on the test set.
    """
    print("Preparing data...")

    # Define the selected features to be used for training
    selected_features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", 
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", 
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", 
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age"
    ]

    # Load the dataset
    data = pd.read_csv('../data/diabetes_indicator.csv')

    # Ensure that the data only includes the selected features
    X = data[selected_features]
    y = data['Diabetes_012']  # Assuming this is your target column

    # Split the data into training and testing sets (80/20 split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, '../scaler.pkl')

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Ensure y_train is a pandas series
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define and train the model
    model = DiabetesModel()  # Define your model class (not shown here)
    
    print("Training model...")
    trained_model = train_model(model, train_loader)  # Ensure `train_model` is defined correctly

    # Evaluate the model
    print("Evaluating model...")
    accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)  # Ensure `evaluate_model` is defined correctly

    return trained_model, scaler, accuracy
