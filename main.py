from models.neural_network import DiabetesModel, train_model, evaluate_model
from utils.preprocess import prepare_data

if __name__ == "__main__":
    # Step 1: Prepare Data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data('data/diabetes_indicator.csv')

    # Step 2: Initialize Model
    model = DiabetesModel()  # Create an instance of the model

    # Step 3: Create DataLoader
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Step 4: Train the Model
    print("Training model...")
    trained_model = train_model(model, train_loader)

    # Step 5: Evaluate the Model
    print("Evaluating model...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    evaluate_model(trained_model, X_test_tensor, y_test)
