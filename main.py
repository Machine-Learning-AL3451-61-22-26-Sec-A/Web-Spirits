import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to load data
@st.cache
def load_data():
    data = pd.read_csv('main.csv')
    return data

# Function to preprocess data
def preprocess_data(data):
    le = LabelEncoder()
    for column in data.columns:
        data[column] = le.fit_transform(data[column])
    return data

# Function to train the model
def train_model(data):
    X = data.drop('PlayTennis', axis=1)
    y = data['PlayTennis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = CategoricalNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

# Load and preprocess data
data = load_data()
preprocessed_data = preprocess_data(data)

# Train model and get accuracy
model, accuracy, X_test, y_test, y_pred = train_model(preprocessed_data)

# Streamlit app
st.title('Naïve Bayesian Classifier')

st.write('### Training Data')
st.write(data)

st.write('### Preprocessed Data')
st.write(preprocessed_data)

st.write('### Model Accuracy')
st.write(f'Accuracy: {accuracy * 100:.2f}%')

st.write('### Test Set Predictions')
test_results = X_test.copy()
test_results['Actual'] = y_test
test_results['Predicted'] = y_pred
st.write(test_results)
