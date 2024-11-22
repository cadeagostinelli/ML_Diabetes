import sys
import os
import tkinter as tk
from tkinter import Toplevel, Label, Button, messagebox
from threading import Thread
import torch
import joblib

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
        global trained_model, scaler
        try:
            # Train and evaluate the model
            trained_model, scaler, accuracy = train_and_evaluate()

            # Save the scaler for normalization during prediction
            joblib.dump(scaler, os.path.join(ROOT_DIR, "scaler.pkl"))

            # Notify user of training completion
            loading_screen.destroy()
            messagebox.showinfo(
                "Training Completed",
                f"Model trained successfully!\nTest Accuracy: {accuracy:.2f}%",
                parent=root
            )
            training_completed = True
            predict_button.config(state=tk.NORMAL)  # Enable prediction button

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
    if not training_completed:
        messagebox.showerror(
            "Error", "Model is not trained yet. Please wait for training to complete.", parent=root
        )
        return

    try:
        # Gather user inputs from GUI fields
        inputs = [float(entry.get()) for entry in entries]
        inputs = torch.tensor(inputs, dtype=torch.float32).reshape(1, -1)

        # Normalize inputs using the saved scaler
        inputs = scaler.transform(inputs)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        # Use the trained model to make predictions
        trained_model.eval()
        with torch.no_grad():
            outputs = trained_model(inputs_tensor)
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

    except ValueError:
        messagebox.showerror(
            "Error", "Please enter valid numeric values for all fields.", parent=root
        )


# Create the Tkinter GUI
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("500x900")

# Display a message indicating training status
Label(root, text="Training the model. Please wait...", font=("Arial", 12)).pack(pady=10)

# Input labels and their descriptions
fields = [
    ("HighBP", "Enter 1 if diagnosed with high blood pressure, otherwise 0"),
    ("HighChol", "Enter 1 if diagnosed with high cholesterol, otherwise 0"),
    ("CholCheck", "Enter 1 if cholesterol was checked in the last 5 years, otherwise 0"),
    ("BMI", "Enter Body Mass Index (e.g., 25.3 for normal weight)"),
    ("Smoker", "Enter 1 if you smoke currently, otherwise 0"),
    ("Stroke", "Enter 1 if you've had a stroke, otherwise 0"),
    ("Fruits", "Enter 1 if you consume fruits at least once daily, otherwise 0"),
    ("Veggies", "Enter 1 if you consume vegetables at least once daily, otherwise 0"),
    ("HvyAlcoholConsump", "Enter 1 if you drink heavily (more than 14 drinks/week for men or 7 for women), otherwise 0"),
    ("AnyHealthcare", "Enter 1 if you have health coverage, otherwise 0"),
    ("NoDocbcCost", "Enter 1 if you couldn’t see a doctor due to cost, otherwise 0"),
    ("GenHlth", "Rate general health (1: Excellent, 5: Poor)"),
    ("MentHlth", "Number of days mental health was not good in the past month"),
    ("PhysHlth", "Number of days physical health was not good in the past month"),
    ("DiffWalk", "Enter 1 if you have difficulty walking or climbing stairs, otherwise 0"),
    ("Sex", "Enter 1 for male, 0 for female"),
    ("Age", "Enter age in years"),
    ("Education", "Enter education level (1: Less than high school, 6: College graduate)"),
    ("Income", "Enter income level (1: Less than $10,000, 8: $75,000 or more)")
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
