import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('insta_train.csv')

# Assuming the last column is the label and the rest are features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Existing Neural Network using Keras
model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Assuming you have a binary classification problem
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
keras_predictions = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Evaluate the Keras model
keras_accuracy = accuracy_score(y_test, keras_predictions)
keras_classification_report = classification_report(y_test, keras_predictions)

# Print the results for the Keras model
print("Keras Model Accuracy:", keras_accuracy)
print("Keras Model Classification Report:")
print(keras_classification_report)

# Additional Neural Network using scikit-learn MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train_scaled, y_train)
mlp_predictions = mlp_classifier.predict(X_test_scaled)

# Evaluate the MLP Classifier
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
mlp_classification_report = classification_report(y_test, mlp_predictions)

# Print the results for the MLP Classifier
print("\nMLP Classifier Accuracy:", mlp_accuracy)
print("MLP Classifier Classification Report:")
print(mlp_classification_report)

# Read the CSV file
file_path = './sampleRealProfile.csv'  # Replace with the actual path to your CSV file
sample_data = pd.read_csv(file_path)
print("Data frame")
print(sample_data.head())

# Additional Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
rf_predictions = rf_classifier.predict(X_test_scaled)

# Evaluate the Random Forest Classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)

# Print the results for the Random Forest Classifier
print("\nRandom Forest Classifier Accuracy:", rf_accuracy)
print("Random Forest Classifier Classification Report:")
print(rf_classification_report)

# Scale the sample data using the same scaler used for training
sample_data_scaled = scaler.transform(sample_data)

# Predict using the Keras model
keras_sample_prediction = (model.predict(sample_data_scaled) > 0.5).astype("int32")
# Predict using the MLP Classifier
mlp_sample_prediction = mlp_classifier.predict(sample_data_scaled)
# Predict using the Random Forest Classifier
rf_sample_prediction = rf_classifier.predict(sample_data_scaled)

if keras_sample_prediction[0] == 0:
    print("These is an original account by Artificial neural network algorithm")
else:
    print("These is a Fake account by Artificial neural network algorithm")

if mlp_sample_prediction[0] == 0:
    print("These is an original account by MLP Classifier")
else:
    print("These is a Fake account by MLP Classifier algorithm")

if rf_sample_prediction[0] == 0:
    print("These is an original account by Random Forest Classifier algorithm")
else:
    print("These is a Fake account by Random Forest Classifier algorithm")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot confusion matrix for Keras Model
plot_confusion_matrix(y_test, keras_predictions, "Keras Model Confusion Matrix")

# Plot confusion matrix for MLP Classifier
plot_confusion_matrix(y_test, mlp_predictions, "MLP Classifier Confusion Matrix")

# Plot confusion matrix for Random Forest Classifier
plot_confusion_matrix(y_test, rf_predictions, "Random Forest Classifier Confusion Matrix")

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred_proba, title):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.show()

# Plot ROC and Precision-Recall curves for Keras Model
keras_pred_proba = model.predict(X_test_scaled)
plot_roc_curve(y_test, keras_pred_proba, "Keras Model ROC Curve")
plot_precision_recall_curve(y_test, keras_pred_proba, "Keras Model Precision-Recall Curve")

# Plot ROC and Precision-Recall curves for MLP Classifier
mlp_pred_proba = mlp_classifier.predict_proba(X_test_scaled)[:, 1]
plot_roc_curve(y_test, mlp_pred_proba, "MLP Classifier ROC Curve")
plot_precision_recall_curve(y_test, mlp_pred_proba, "MLP Classifier Precision-Recall Curve")

# Plot ROC and Precision-Recall curves for Random Forest Classifier
rf_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]
plot_roc_curve(y_test, rf_pred_proba, "Random Forest Classifier ROC Curve")
plot_precision_recall_curve(y_test, rf_pred_proba, "Random Forest Classifier Precision-Recall Curve")

import numpy as np

# Function to calculate AUC for multiple models
def calculate_auc(models, X_test_scaled_list, y_test):
    auc_scores = []

    for model, X_test_scaled in zip(models, X_test_scaled_list):
        y_pred_proba = model.predict(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    return auc_scores

# List of models and their corresponding test data
models = [model, mlp_classifier, rf_classifier]
X_test_scaled_list = [X_test_scaled, X_test_scaled, X_test_scaled]
model_names = ['Keras Model', 'MLP Classifier', 'Random Forest Classifier']

# Calculate AUC scores
auc_scores = calculate_auc(models, X_test_scaled_list, y_test)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(model_names, auc_scores, color=['blue', 'orange', 'green'])
plt.xlabel('Models')
plt.ylabel('AUC Score')
plt.title('Comparison of AUC Scores for Keras, MLP, and Random Forest')
plt.show()