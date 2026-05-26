import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# Step 1: Data Collection
# ==========================================
# Loading the Wisconsin Breast Cancer Dataset (Available on Kaggle)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = Malignant, 1 = Benign

# ==========================================
# Step 2: Preprocessing
# ==========================================
# Checking for missing values (This dataset is pre-cleaned, but standard practice applies)
X.fillna(X.mean(), inplace=True)

# Splitting the dataset into training (80%) and testing/evaluation (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data using StandardScaler
# Neural networks perform best when input features are on a similar scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Step 3: Model Building
# ==========================================
# Initializing the Artificial Neural Network (ANN)
model = Sequential()

# Input Layer and First Hidden Layer (30 features input)
# Using ReLU activation to introduce non-linearity
model.add(Dense(units=16, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.2)) # Step 6: Dropout added for improvement (prevents overfitting)

# Second Hidden Layer
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))

# Output Layer (Binary classification: Malignant or Benign)
# Sigmoid activation outputs a probability between 0 and 1
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
# Adam optimizer is used for adaptive learning rate; Binary Crossentropy is standard for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# Step 4: Training
# ==========================================
# Training the model with the preprocessed data
# Using a validation split to monitor performance on unseen data during training
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1 # Set to 0 to silence training text output
)

# ==========================================
# Step 5: Evaluation
# ==========================================
# Predicting the test set results
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculating Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)