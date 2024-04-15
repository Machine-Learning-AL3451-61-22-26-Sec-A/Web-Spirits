import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Define the Streamlit app
def main():
    st.title("ID3 Algorithm Demo")

    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Display the dataset
    st.subheader("Iris Dataset")
    st.write(X)

    # Create a decision tree model
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)

    # Display the decision tree rules
    st.subheader("Decision Tree Rules")
    rules = export_text(clf, feature_names=iris.feature_names)
    st.text_area("Decision Tree Rules", rules, height=300)

    # Make a prediction
    st.subheader("Make Prediction")
    input_features = {}
    for feature in X.columns:
        input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

    if st.button("Predict"):
        instance = pd.DataFrame([input_features])
        prediction = clf.predict(instance)
        st.success(f"The predicted class is {iris.target_names[prediction[0]]}")

# Run the app
if __name__ == "__main__":
    main()
