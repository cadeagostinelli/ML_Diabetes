# Diabetes Risk Prediction with Machine Learning

This project predicts the risk of diabetes (No Diabetes, Pre-Diabetes, or Diabetes) using a machine learning model. It includes an intuitive graphical user interface (GUI) to enable easy interaction and real-time predictions based on health metrics.

---

## Overview

Diabetes is a growing health concern worldwide. Early identification of diabetes risk can help in taking preventive actions. This project utilizes a Neural Network model trained on a dataset of health indicators to classify individuals into three risk categories: No Diabetes, Pre-Diabetes, or Diabetes. The project integrates machine learning techniques with a user-friendly interface to make this solution accessible for non-technical users.

---

## Features

- **Machine Learning Model:** A neural network built with PyTorch to predict diabetes risk.
- **Data Preprocessing:** Includes feature normalization and handling of class imbalance.
- **GUI:** A Tkinter-based user interface for easy interaction and prediction.
- **Real-time Results:** Users can input their health metrics and get an instant risk prediction.

---

## Installation

### Prerequisites

1. Python 3.8 or higher installed on your system.
2. Git installed (optional for cloning the repository).

### Steps

1. Download the Files:  
   Download the provided `.zip` file from the submission or copy all project files to your local machine.
   Extract the `.zip` file (if applicable).

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   ```
3. Install the dependencies:
   Open a terminal or command prompt, navigate to the project directory, and run the following command to install all required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. File Structure
   Place the dataset file diabetes_indicator.csv in the data/ folder within the project directory. This file contains the health metrics used for training the model.
   '''ML_Diabetes/
   │
   ├── data/
   │ └── diabetes_indicator.csv # Dataset
   ├── models/
   │ └── neural_network.py # Neural network model and training logic
   ├── utils/
   │ └── preprocess.py # Data preprocessing logic
   ├── app.py # GUI application
   ├── main.py # Training and evaluation logic
   ├── requirements.txt # Python dependencies
   └── README.md # Project documentation
   '''

---

## Running the Application

1. **Ensure the Dataset is Available:**  
   Place the `diabetes_indicator.csv` dataset in the `data/` directory.

2. **Run the Application:**  
    Start the GUI by running the `app.py` file:
   ```bash
   python app.py
   ```
   - The application will begin by training the neural network model.
   - Once the training is complete, the prediction feature will be enabled.

## Implementation

- When using the application, you will be prompted to fill in several health-related metrics. Input the values based on the prompted metrics.
- The application will then reliably classifies users into three categories: No Diabetes, Pre-Diabetes, or Diabetes.

---

### Project Workflow

1. Data Preprocessing:

   - Features from the dataset are scaled using StandardScaler to ensure consistent ranges across all input variables.
   - The dataset is split into training (80%) and testing (20%) subsets.

2. Model Training:

   - A three-layer neural network with ReLU activation is used.
   - The model predicts one of three classes: No Diabetes, Pre-Diabetes, or Diabetes.
   - Class weights are applied to address data imbalance, ensuring equal attention to all categories.

3. Prediction:
   - Users provide health metrics via the GUI.
   - Inputs are validated and normalized before being fed into the trained model.
   - The model outputs a risk classification, which is displayed in a pop-up window.

---

## Acknowledgements

    This project was developed to combine machine learning expertise with user-centered design. Ultimately, we learned data preprocessing, feature scaling, and class imbalance management to develop and implement a healthcare neural network. We improved our grasp of machine learning-expert system integration to ensure interpretability and accuracy. Most significantly, we recognized that user-centric design makes technology accessible and effective in solving real-world problems.

---

### THANK YOU

----------x----------x----------x---------x----------x----------x----------x---------x----------x----------x----------x---------x----------x----------x----------x---------x----------x----------x----------x---------x----------x----------x----------x---------x----------x----------x----------x---------x
