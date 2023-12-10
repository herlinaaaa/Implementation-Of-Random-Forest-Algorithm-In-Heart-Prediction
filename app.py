from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset and preprocess as you did before
df = pd.read_csv("heart.csv")
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

# Separate features and target variable
y = df['output']
X = df.drop(['output'], axis=1)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=101, stratify=y, test_size=0.25)

# Train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Function to take user input and make predictions
def predict_heart_disease(user_input):
    user_data = pd.DataFrame([user_input], columns=X_train.columns)
    user_data[columns_to_scale] = standardScaler.transform(user_data[columns_to_scale])
    prediction = rf.predict(user_data)
    return "More Chance of Heart Attack." if prediction[0] == 1 else "Less Chance of Heart Attack."

@app.route('/')
def index():
    return render_template('index.html', features=X_train.columns)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = [float(request.form[key]) for key in X_train.columns]
        result = predict_heart_disease(user_input)
        return render_template('index.html', features=X_train.columns, result=result)

if __name__ == '__main__':
    app.run(debug=True)
