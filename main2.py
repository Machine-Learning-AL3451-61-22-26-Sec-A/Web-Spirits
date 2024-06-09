import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import graphviz

st.write("Team: 22AIA-WEBSPIRITS")
st.title("Decision Tree (ID3) Algorithm Demonstration")

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

def build_and_train_decision_tree(data, target_column):
    features = data.drop(target_column, axis=1)
    target = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy, features.columns

def visualize_tree(clf, feature_names):
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_names,  
                                    class_names=clf.classes_,  
                                    filled=True, rounded=True,  
                                    special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return graph

st.write("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Training Data:")
    st.write(data)
    
    target_column = st.selectbox("Select the target column", data.columns)
    
    if target_column:
        clf, accuracy, feature_names = build_and_train_decision_tree(data, target_column)
        
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        st.write("Decision Tree Visualization:")
        graph = visualize_tree(clf, feature_names)
        st.graphviz_chart(graph)
        
        st.write("Classify a new sample")
        new_sample = {}
        for feature in feature_names:
            new_sample[feature] = st.text_input(f"Enter value for {feature}")
        
        if st.button("Classify"):
            sample_df = pd.DataFrame([new_sample])
            prediction = clf.predict(sample_df)
            st.write(f"Predicted class: {prediction[0]}")
