import tkinter as tk
from tkinter import messagebox
import joblib
from models.model_utils import load_model
from utils.preprocess import preprocess_data


def submit():
    try:
        input_data = [float(entry.get()) for entry in entries]
        scaler = joblib.load('scaler.pkl')
        input_data_scaled = scaler.transform([input_data])
        model = load_model('diabetes_model.pth')
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        risk_levels = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
        messagebox.showinfo("Prediction", f"Diabetes Risk Level: {risk_levels[predicted.item()]}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric inputs!")


# GUI Setup
root = tk.Tk()
root.title("Diabetes Risk Prediction")
labels = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Fruits", "Veggies", "HvyAlcoholConsump",
          "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
entries = []

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

tk.Button(root, text="Submit", command=submit).grid(row=len(labels), column=0, columnspan=2)
root.mainloop()
