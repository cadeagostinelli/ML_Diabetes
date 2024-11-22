import torch
from models.neural_network import DiabetesModel, train_model, evaluate_model
from utils.preprocess import prepare_data
import joblib


def train_and_evaluate():
    """
    Function to train the model and evaluate its performance.
    Returns:
        trained_model: The trained PyTorch model.
        accuracy: Accuracy of the model on the test set.
    """
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('data/diabetes_indicator.csv')

    model = DiabetesModel()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("Training model...")
    trained_model = train_model(model, train_loader)

    print("Evaluating model...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    accuracy = evaluate_model(trained_model, X_test_tensor, y_test)

    scaler = joblib.load('scaler.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return trained_model, accuracy
