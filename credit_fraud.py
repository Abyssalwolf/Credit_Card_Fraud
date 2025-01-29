import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# App Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Models")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.selectbox(
    "Choose a section:",
    ("Pre-Processing", "Model Results"),
    index=0
)

if section == "Pre-Processing":
    st.header("Data Pre-Processing")
    st.markdown(
        "### Insights and Graphs\n"
        "Explore how the dataset is prepared for machine learning models."
    )

    # Load dataset from the same directory
    try:
        # Ensure that the dataset file matches the code reference
        # If your dataset has a different name, change both this line and the error message accordingly
        df = pd.read_csv("creditcard.csv")

        st.write("### Dataset Overview")
        st.dataframe(df.head())

        st.markdown(
            "#### Explanation\n"
            "The dataset contains the following columns:\n"
            "- `Time`: Time elapsed between each transaction and the first transaction.\n"
            "- `V1` to `V28`: Principal components (features after PCA transformation).\n"
            "- `Amount`: Transaction amount.\n"
            "- `Class`: Target variable (1 for fraud, 0 for non-fraud)."
        )

        # Class distribution with percentages
        st.subheader("Class Distribution")
        st.markdown(
            "This plot shows the imbalance in the dataset, with percentages to highlight the fraud vs. non-fraud ratio."
        )
        class_counts = df['Class'].value_counts()
        class_percentages = (class_counts / len(df)) * 100
        fig, ax = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_percentages, palette="pastel", ax=ax)
        ax.set_title("Class Distribution (Fraud vs. Non-Fraud)")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        ax.set_ylabel("Percentage")
        for i, v in enumerate(class_percentages):
            ax.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        st.pyplot(fig)

        st.subheader("Distributions")
        st.markdown(
            "By seeing the distributions we can have an idea how skewed these features are."
        )
        fig, ax = plt.subplots(1, 2, figsize=(18, 4))
        amount_val = df['Amount'].values
        time_val = df['Time'].values

        # Plot the distribution of transaction amounts using histplot
        sns.histplot(amount_val, kde=True, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlabel('Transaction Amount')
        ax[0].set_ylabel('Frequency')

        # Plot the distribution of transaction times using histplot
        sns.histplot(time_val, kde=True, ax=ax[1], color='b')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlabel('Transaction Time')
        ax[1].set_ylabel('Frequency')

        # Render the plot in Streamlit
        st.pyplot(fig)

        st.subheader("Scaling and (Optional) Balancing")
        st.markdown(
            "We will first scale the Time and Amount columns (just like the other columns). "
            "Optionally, we demonstrate a simple under-sampling approach to address the class imbalance."
        )

        # Scaling Time and Amount
        rob_scaler = RobustScaler()
        df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

        # Drop original columns
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        st.write("Scaled `Time` and `Amount` features (first 5 rows):")
        st.dataframe(df[['scaled_time', 'scaled_amount']].head())

        # Simple Under-Sampling (Optional)
        # This code reduces the "non-fraud" class to match the number of "fraud" records
        # to achieve a balanced ratio. Remove or modify if you prefer other methods.
        fraud_df = df[df['Class'] == 1]
        non_fraud_df = df[df['Class'] == 0]

        # Under-sampling: randomly select from non-fraud, the same count as fraud
        non_fraud_sampled = non_fraud_df.sample(n=len(fraud_df), random_state=42)
        balanced_df = pd.concat([fraud_df, non_fraud_sampled], axis=0).sample(frac=1, random_state=42)

        st.markdown(
            "#### After Under-Sampling\n"
            "We now have a balanced dataset for training. We shuffle to remove ordering biases."
        )
        class_counts_balanced = balanced_df['Class'].value_counts()
        st.write(class_counts_balanced)

        # Prepare features and targets for modeling
        X = balanced_df.drop('Class', axis=1)
        y = balanced_df['Class']

        # StratifiedKFold with a set random_state for reproducible splits
        sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for train_index, test_index in sss.split(X, y):
            original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
            original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
            # Break after the first split if you only want a single train/test
            break

        # Convert to NumPy arrays (optional, depending on your preference)
        original_Xtrain = original_Xtrain.values
        original_Xtest = original_Xtest.values
        original_ytrain = original_ytrain.values
        original_ytest = original_ytest.values

        # Save in session_state for later steps
        st.session_state['X_train'] = original_Xtrain
        st.session_state['X_test'] = original_Xtest
        st.session_state['y_train'] = original_ytrain
        st.session_state['y_test'] = original_ytest

        st.write('-' * 100)

    except FileNotFoundError:
        st.error("Dataset file not found in the directory. Please ensure 'creditcard.csv' is available.")

elif section == "Model Results":
    # Sidebar for model selection
    st.sidebar.header("Explore Models")
    st.sidebar.markdown(
        "Select a machine learning model to visualize its performance in detecting credit card fraud."
    )
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Logistic Regression","Naive Bayes"),
        index=0
    )

    if 'X_train' not in st.session_state:
        st.error("Please preprocess the data in the 'Pre-Processing' section first.")
    else:
        # Load preprocessed data
        original_Xtrain = st.session_state['X_train']
        original_Xtest = st.session_state['X_test']
        original_ytrain = st.session_state['y_train']
        original_ytest = st.session_state['y_test']

    # Placeholder for displaying content based on model choice
    st.header(f"Results: {model_choice}")
    st.markdown(
        "### Model Performance Summary\n"
        "Below are the results of the selected machine learning model applied to detect credit card fraud."
    )

    if 'X_train' in st.session_state:
        if model_choice == "K-Nearest Neighbors (KNN)":
            st.subheader("KNN Performance")
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = knn.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Support Vector Machine (SVM)":
            st.subheader("SVM Performance")
            # Train SVM with a set random_state for reproducibility
            svm = SVC(kernel='linear', probability=True, random_state=42)
            svm.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = svm.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Logistic Regression":
            st.subheader("Logistic Regression Performance")
            # Train Logistic Regression
            # Set a higher max_iter and a random_state for reproducibility
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = log_reg.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Classification Report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix Heatmap
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Oranges",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_choice == "Naive Bayes":
            st.subheader("Naive Bayes Performance")
            # Train the model
            nb_model = GaussianNB()
            nb_model.fit(original_Xtrain, original_ytrain)

            # Predictions
            y_pred = nb_model.predict(original_Xtest)

            # Metrics
            accuracy = accuracy_score(original_ytest, y_pred)
            report = classification_report(original_ytest, y_pred, output_dict=True)
            confusion = confusion_matrix(original_ytest, y_pred)

            # Display accuracy
            st.write(f"### Accuracy: **{accuracy * 100:.2f}%**")

            # Display classification report
            st.write("#### Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            # Display confusion matrix
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Purples",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        # Footer or additional notes
        st.markdown("\n---\n")
        st.markdown(
            "This application provides a comparative analysis of different machine learning models for credit card fraud detection. "
            "Use the sidebar to switch between models and view their results in an easy-to-understand format."
        )