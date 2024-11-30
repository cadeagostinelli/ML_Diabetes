import sys
import os
import tkinter as tk
from tkinter import Toplevel, Label, Button, messagebox
from threading import Thread
import torch
import joblib
import pandas as pd

# Add the root directory (ML_Diabetes/) to Python's module path dynamically
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import model and training function
from models.neural_network import DiabetesModel
from main import train_and_evaluate  # Ensure main.py is in the root folder

# Global variables for the trained model, scaler, and status
trained_model = None
scaler = None
training_completed = False


# Function to display a loading screen
def show_loading_screen(message="Processing..."):
    loading_screen = Toplevel(root)
    loading_screen.title("Loading")
    loading_screen.geometry("400x150")
    Label(loading_screen, text=message, font=("Arial", 12)).pack(pady=20)
    return loading_screen


# Function to train the model
def train_model_ui():
    global trained_model, scaler, training_completed

    # Show the loading screen
    loading_screen = show_loading_screen("Training the model, please wait...")

    # Training function to run in a background thread
    def train():
        global trained_model, scaler, training_completed
        try:
            # Train and evaluate the model
            trained_model, scaler, accuracy = train_and_evaluate()

            # Save the scaler for normalization during prediction
            joblib.dump(scaler, os.path.join(ROOT_DIR, "../scaler.pkl"))

            # Save the trained model
            torch.save(trained_model.state_dict(), os.path.join(ROOT_DIR, "diabetes_model.pth"))

            # Notify user of training completion
            training_completed = True  # Update training status here
            predict_button.config(state=tk.NORMAL)  # Enable prediction button

            # Destroy loading screen and show completion message
            loading_screen.destroy()
            messagebox.showinfo(
                "Training Completed",
                f"Model trained successfully!\nTest Accuracy: {accuracy:.2f}%",
                parent=root
            )

        except Exception as e:
            loading_screen.destroy()
            messagebox.showerror(
                "Error", f"An error occurred during training: {e}", parent=root
            )
            sys.exit(1)

    # Run the training function in a separate thread
    thread = Thread(target=train)
    thread.start()


# Function to predict diabetes risk
def predict_diabetes():
    global trained_model, training_completed

    if not training_completed:
        messagebox.showerror(
            "Error", "The model has not finished training. Please wait.", parent=root
        )
        return

    try:
                # Gather and validate user inputs
        inputs = []
        invalid_fields = []
        for i, (entry, (field_name, _)) in enumerate(zip(entries, fields)):
            value = entry.get()
            try:
                # Attempt to convert the input to a float
                inputs.append(float(value))
            except ValueError:
                # Collect the name of the field with invalid input
                invalid_fields.append(field_name)

        # If there are any invalid fields, show an error message and stop
        if invalid_fields:
            invalid_fields_str = "\n".join(invalid_fields)
            messagebox.showerror(
                "Input Error",
                f"The following fields have invalid or missing values:\n{invalid_fields_str}",
                parent=root
            )
            return

        # Convert validated inputs to a PyTorch tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).reshape(1, 19)  # Ensure it has 21 features

        # Convert inputs_tensor to a DataFrame with appropriate column names
        input_df = pd.DataFrame(inputs_tensor.numpy(), columns=[field[0] for field in fields])

        # Normalize using the scaler
        inputs_scaled = scaler.transform(input_df)
        inputs_scaled_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)


        # Load the trained model
        model_path = os.path.join(ROOT_DIR, "diabetes_model.pth")
        trained_model.load_state_dict(torch.load(model_path, weights_only=True))
        trained_model.eval()

        # Use the trained model to make predictions
        with torch.no_grad():
            outputs = trained_model(inputs_scaled_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map predictions to risk levels
        risk_levels = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
        result = risk_levels[predicted.item()]

        # Show result in a larger pop-up
        result_popup = Toplevel(root)
        result_popup.title("Prediction Result")
        result_popup.geometry("400x200")
        Label(result_popup, text=f"Diabetes Risk: {result}", font=("Arial", 16), fg="blue").pack(pady=40)

        Button(result_popup, text="Close", command=result_popup.destroy, font=("Arial", 12)).pack(pady=20)

    except Exception as e:
        messagebox.showerror(
            "Error", f"An unexpected error occurred: {e}", parent=root
        )





# Create the Tkinter GUI
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("500x900")

# Display a message indicating training status
Label(root, text="Training the model. Please wait...", font=("Arial", 12)).pack(pady=10)

# Input labels and their descriptions
fields = [
    ("HighBP", "1 if high blood pressure, 0 otherwise"),
    ("HighChol", "1 if high cholesterol, 0 otherwise"),
    ("CholCheck", "1 if cholesterol checked recently, 0 otherwise"),
    ("BMI", "Body Mass Index (e.g., 25.3)"),
    ("Smoker", "1 if you smoke, 0 otherwise"),
    ("Stroke", "1 if you've had a stroke, 0 otherwise"),
    ("HeartDiseaseorAttack", "1 if heart disease/attack, 0 otherwise"),
    ("PhysActivity", "1 if regular physical activity, 0 otherwise"),
    ("Fruits", "1 if daily fruit intake, 0 otherwise"),
    ("Veggies", "1 if daily vegetable intake, 0 otherwise"),
    ("HvyAlcoholConsump", "1 if heavy alcohol consumption, 0 otherwise"),
    ("AnyHealthcare", "1 if health coverage, 0 otherwise"),
    ("NoDocbcCost", "1 if couldn't see doctor due to cost, 0 otherwise"),
    ("GenHlth", "Rate health (1: Excellent, 5: Poor)"),
    ("MentHlth", "Days mental health not good (past month)"),
    ("PhysHlth", "Days physical health not good (past month)"),
    ("DiffWalk", "1 if difficulty walking, 0 otherwise"),
    ("Sex", "1 for male, 0 for female"),
    ("Age", "Enter age"),
]

entries = []

# Input fields with descriptions
for field, description in fields:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=field, width=20, anchor="w").pack(side="left")
    entry = tk.Entry(frame, width=20)
    entry.pack(side="right")
    entries.append(entry)

    # Add the description below the input field
    desc_label = tk.Label(frame, text=description, wraplength=400, fg="gray", font=("Arial", 8))
    desc_label.pack(pady=2)

# Predict Button (initially disabled)
predict_button = Button(root, text="Predict Diabetes Risk", command=predict_diabetes, bg="lightgreen", width=20, state=tk.DISABLED)
predict_button.pack(pady=20)

# Start the training process when the app launches
train_model_ui()

# Run the Tkinter GUI main loop
root.mainloop()
