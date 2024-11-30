import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(filepath, test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Select only the required columns
    X = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
            'Sex', 'Age', 'Education', 'Income']].values
    y = df['Diabetes_012'].values

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
