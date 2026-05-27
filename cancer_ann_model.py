"""
Artificial Neural Network (ANN) for Cancer Diagnosis using TensorFlow
Classifies breast cancer as malignant or benign using radiological data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import config
from data_pipeline import DataPipeline
import os


class CancerANNModel:
    """Artificial Neural Network model for cancer diagnosis"""

    def __init__(self):
        """Initialize the model"""
        self.model = None
        self.history = None
        self.predictions = None
        self.metrics = {}

    def build_model(self, input_dim):
        """
        Build the ANN model architecture

        Args:
            input_dim (int): Number of input features
        """
        print("\n" + "="*60)
        print("BUILDING ANN MODEL")
        print("="*60)

        self.model = Sequential()

        # Input layer + Hidden layer 1
        self.model.add(
            Dense(
                config.HIDDEN_LAYER_1_UNITS,
                activation='relu',
                input_shape=(input_dim,),
                name='hidden_layer_1'
            )
        )
        self.model.add(Dropout(config.DROPOUT_RATE, name='dropout_1'))

        # Hidden layer 2
        self.model.add(
            Dense(
                config.HIDDEN_LAYER_2_UNITS,
                activation='relu',
                name='hidden_layer_2'
            )
        )
        self.model.add(Dropout(config.DROPOUT_RATE, name='dropout_2'))

        # Output layer
        self.model.add(
            Dense(1, activation='sigmoid', name='output_layer')
        )

        print("Model Architecture:")
        print(f"  - Input Features: {input_dim}")
        print(f"  - Hidden Layer 1: {config.HIDDEN_LAYER_1_UNITS} neurons (ReLU) + Dropout({config.DROPOUT_RATE})")
        print(f"  - Hidden Layer 2: {config.HIDDEN_LAYER_2_UNITS} neurons (ReLU) + Dropout({config.DROPOUT_RATE})")
        print(f"  - Output Layer: 1 neuron (Sigmoid)")

    def compile_model(self):
        """Compile the model"""
        print("\n" + "="*60)
        print("COMPILING MODEL")
        print("="*60)

        self.model.compile(
            optimizer=config.OPTIMIZER,
            loss=config.LOSS_FUNCTION,
            metrics=config.METRICS
        )

        print(f"Optimizer: {config.OPTIMIZER}")
        print(f"Loss Function: {config.LOSS_FUNCTION}")
        print(f"Metrics: {config.METRICS}")

        # Display model summary
        print("\nModel Summary:")
        self.model.summary()

    def train_model(self, X_train, y_train):
        """
        Train the ANN model

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=config.RESTORE_BEST_WEIGHTS,
            verbose=1
        )

        print(f"Training Parameters:")
        print(f"  - Epochs: {config.EPOCHS}")
        print(f"  - Batch Size: {config.BATCH_SIZE}")
        print(f"  - Validation Split: {config.VALIDATION_SPLIT*100:.0f}%")
        print(f"  - Early Stopping Enabled (patience={config.EARLY_STOPPING_PATIENCE})")

        # Train the model
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            callbacks=[early_stop],
            verbose=config.VERBOSE
        )

        print("\nTraining completed!")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Get loss and accuracy
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Get predictions
        self.predictions = self.model.predict(X_test)
        self.predictions_binary = (self.predictions > 0.5).astype(int).flatten()

        # Calculate metrics
        self.metrics['accuracy'] = accuracy_score(y_test, self.predictions_binary)
        self.metrics['precision'] = precision_score(y_test, self.predictions_binary)
        self.metrics['recall'] = recall_score(y_test, self.predictions_binary)
        self.metrics['f1_score'] = f1_score(y_test, self.predictions_binary)

        print("\nDetailed Metrics:")
        print(f"  - Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  - Precision: {self.metrics['precision']:.4f}")
        print(f"  - Recall:    {self.metrics['recall']:.4f}")
        print(f"  - F1-Score:  {self.metrics['f1_score']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, self.predictions_binary,
                                   target_names=['Benign (0)', 'Malignant (1)']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, self.predictions_binary)
        print(cm)

        return self.metrics

    def save_model(self):
        """Save the trained model"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)

        self.model.save(config.MODEL_PATH)
        print(f"Model saved to: {config.MODEL_PATH}")

    def save_metrics(self, y_test):
        """Save evaluation metrics to file"""
        print("\nSaving metrics...")

        with open(config.METRICS_OUTPUT_PATH, 'w') as f:
            f.write("CANCER ANN MODEL - EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Accuracy:  {self.metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {self.metrics['precision']:.4f}\n")
            f.write(f"Recall:    {self.metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {self.metrics['f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, self.predictions_binary,
                                         target_names=['Benign (0)', 'Malignant (1)']))

        print(f"Metrics saved to: {config.METRICS_OUTPUT_PATH}")

    def save_predictions(self, X_test):
        """Save predictions to CSV file"""
        print("Saving predictions...")

        predictions_df = pd.DataFrame({
            'predicted_probability': self.predictions.flatten(),
            'predicted_class': self.predictions_binary
        })

        predictions_df.to_csv(config.PREDICTIONS_OUTPUT_PATH, index=False)
        print(f"Predictions saved to: {config.PREDICTIONS_OUTPUT_PATH}")

    def plot_training_history(self):
        """Plot training and validation accuracy/loss"""
        print("\nGenerating training history plots...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {os.path.join(config.RESULTS_DIR, 'training_history.png')}")
        plt.close()

    def plot_confusion_matrix(self, y_test):
        """Plot confusion matrix"""
        print("Generating confusion matrix plot...")

        cm = confusion_matrix(y_test, self.predictions_binary)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')}")
        plt.close()

    def run_pipeline(self):
        """Execute the complete ANN pipeline"""
        print("\n" + "="*70)
        print("CANCER ANN MODEL - COMPLETE PIPELINE")
        print("="*70)

        try:
            # Data preprocessing
            print("\n[STEP 1] Data Preprocessing")
            pipeline = DataPipeline()
            X_train, X_test, y_train, y_test = pipeline.run_pipeline()

            # Model building
            print("\n[STEP 2] Model Building")
            self.build_model(input_dim=X_train.shape[1])

            # Model compilation
            print("\n[STEP 3] Model Compilation")
            self.compile_model()

            # Model training
            print("\n[STEP 4] Model Training")
            self.train_model(X_train, y_train)

            # Model evaluation
            print("\n[STEP 5] Model Evaluation")
            self.evaluate_model(X_test, y_test)

            # Model improvement (visualization)
            print("\n[STEP 6] Model Visualization")
            self.plot_training_history()
            self.plot_confusion_matrix(y_test)

            # Save model and results
            print("\n[STEP 7] Saving Results")
            self.save_model()
            self.save_metrics(y_test)
            self.save_predictions(X_test)

            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nResults saved to: {config.RESULTS_DIR}")
            print(f"Model saved to: {config.MODEL_PATH}")

        except Exception as e:
            print(f"\nError during pipeline execution: {str(e)}")
            raise


if __name__ == "__main__":
    # Create and run the model
    ann_model = CancerANNModel()
    ann_model.run_pipeline()
