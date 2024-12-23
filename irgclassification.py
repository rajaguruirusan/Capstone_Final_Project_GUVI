import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
#import joblib

# Set up Streamlit app with a wide layout
st.set_page_config(layout="wide")
st.title("GUVI Final Project : Classification | Prediction - IRG")

# Load dataset
data = pd.read_excel(r'C:\Users\Rajaguru Irusan\Documents\DS_Project\Final_Project\Classification\classification_sampledata.xlsx')

# Encode categorical data for prediction
def encode_device_category(category):
    return {
        "Desktop": [1, 0, 0],
        "Mobile": [0, 1, 0],
        "Tablet": [0, 0, 0]
    }.get(category, [0, 0, 0])

# Tabs for the application
tabs = st.tabs(["EDA", "Model Comparison", "Live Prediction"])

# EDA Tab
with tabs[0]:
    st.header("Exploratory Data Analysis")
    eda_option = st.selectbox("Select Plot Type", ["Bar Plot", "Histogram", "Correlation Matrix", "Box Plot"])

    if eda_option == "Bar Plot":
        st.bar_chart(data["has_converted"].value_counts())

    elif eda_option == "Histogram":
        column = st.selectbox("Select Column for Histogram", data.columns)
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=20, color='green', alpha=0.7)
        st.pyplot(fig)

    elif eda_option == "Correlation Matrix":
        st.write("### Correlation Matrix")
        numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
        sns.heatmap(
            numeric_data.corr(), 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True, 
            annot_kws={"size": 8},  # Reduced annotation font size
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)  # Rotate x-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)  # Adjust y-axis label font size
        st.pyplot(fig)

    elif eda_option == "Box Plot":
        column = st.selectbox("Select Column for Box Plot", data.columns)
        fig, ax = plt.subplots()
        sns.boxplot(x="has_converted", y=column, data=data, ax=ax)
        st.pyplot(fig)

# Preprocessing
data.fillna(0, inplace=True)
if "device_deviceCategory" in data.columns:
    data = pd.get_dummies(data, columns=["device_deviceCategory"], drop_first=True)

# Prepare data for modeling
X = data[["count_session", "count_hit"] + [col for col in data.columns if col.startswith("device_deviceCategory_")]]
y = data["has_converted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Comparison Tab
with tabs[1]:
    st.header("Model Comparison")
    models = {
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", solver="saga", max_iter=1000),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced")
    }

    model_metrics = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        accuracy = accuracy_score(y_test, predictions)
        model_metrics[model_name] = {
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"],
            "Accuracy": accuracy
        }

    metrics_df = pd.DataFrame(model_metrics).transpose()
    st.write(metrics_df)

    # Accuracy Visualization
    st.subheader("Model Metrics Visualization")

    # Create two columns
    col1, col2 = st.columns(2)

    # Accuracy Plot
    with col1:
        st.write("### Model Accuracy")
        fig, ax = plt.subplots(figsize=(4, 2.5))  # Smaller figure size
        ax.bar(metrics_df.index, metrics_df["Accuracy"], color='skyblue', label="Accuracy")
        ax.set_ylabel("Score", fontsize=8)  # Smaller font size for labels
        ax.set_title("Model Accuracy", fontsize=10)  # Smaller font size for title
        ax.tick_params(axis='x', labelsize=7)  # Smaller x-axis labels
        ax.tick_params(axis='y', labelsize=7)  # Smaller y-axis labels
        plt.tight_layout()  # Adjust layout to reduce whitespace
        st.pyplot(fig)

    # Precision and Recall Plot
    with col2:
        st.write("### Precision and Recall Comparison")
        fig, ax = plt.subplots(figsize=(4, 2.5))  # Smaller figure size
        width = 0.35  # Bar width
        indices = range(len(metrics_df.index))
        ax.bar(indices, metrics_df["Precision"], width, label="Precision", color='green')
        ax.bar([i + width for i in indices], metrics_df["Recall"], width, label="Recall", color='orange')

        ax.set_ylabel("Score", fontsize=8)  # Smaller font size for labels
        ax.set_title("Precision and Recall", fontsize=10)  # Smaller font size for title
        ax.set_xticks([i + width / 2 for i in indices])
        ax.set_xticklabels(metrics_df.index, fontsize=7)  # Smaller font size for x-axis labels
        ax.legend(fontsize=8)  # Smaller font size for legend
        plt.tight_layout()  # Adjust layout to reduce whitespace
        st.pyplot(fig)

# Live Prediction Tab
with tabs[2]:
    st.header("Live Prediction")
    st.subheader("Enter Prediction Inputs")
    count_session = st.text_input("Enter Session Count", "1")
    count_hit = st.text_input("Enter Hit Count", "1")
    device_deviceCategory = st.selectbox("Device Category", ["Mobile", "Desktop", "Tablet"])

    if st.button("Predict Conversion"):
        encoded_category = encode_device_category(device_deviceCategory)
        live_data = pd.DataFrame({
            "count_session": [float(count_session)],
            "count_hit": [float(count_hit)],
            "device_deviceCategory_Desktop": [encoded_category[0]],
            "device_deviceCategory_Mobile": [encoded_category[1]],
            "device_deviceCategory_Tablet": [encoded_category[2]]
        })

        # Align columns with training data
        for col in X_train.columns:
            if col not in live_data.columns:
                live_data[col] = 0  # Add missing columns with default value 0
        live_data = live_data[X_train.columns]  # Ensure column order matches
        live_data_scaled = scaler.transform(live_data)  # Scale live data

        selected_model = st.selectbox("Select Model for Prediction", list(models.keys()))

        try:
            # Attempt prediction
            prediction = models[selected_model].predict(live_data_scaled)
            probability = models[selected_model].predict_proba(live_data_scaled)[0][1]
            st.write("Prediction:", "Converted" if prediction[0] == 1 else "Not Converted")
            st.write("Conversion Probability:", f"{probability:.2f}")
        except Exception as e:
            # Show error message and fallback to Random Forest
            st.error(f"An error occurred with {selected_model}: {str(e)}")
            st.warning("Falling back to Random Forest for prediction.")
            try:
                prediction = models["Random Forest"].predict(live_data_scaled)
                probability = models["Random Forest"].predict_proba(live_data_scaled)[0][1]
                st.write("Prediction (Random Forest):", "Converted" if prediction[0] == 1 else "Not Converted")
                st.write("Conversion Probability (Random Forest):", f"{probability:.2f}")
            except Exception as fallback_error:
                st.error(f"Fallback to Random Forest also failed: {str(fallback_error)}")
