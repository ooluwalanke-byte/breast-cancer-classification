import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

def build_optimized_network(input_shape, num_classes):
    """
    Constructs the optimized, regularized neural network architecture 
    tailored for Otomoto's customer segmentation metrics.
    """
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),  # Stabilizes input distribution across mini-batches
        Dropout(0.2),          # Prevents co-adaptation of hidden units
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax') # Multi-class probability layout
    ])
    return model

# ==========================================
# SGDM with Nesterov Accelerated Gradient
# ==========================================
# Configuration parameters
INPUT_FEATURES = 30  # Number of input features
CLASSES = 3          # Number of output classes

# Build model
model_sgdm = build_optimized_network(INPUT_FEATURES, CLASSES)

# Initialize SGD optimizer with Nesterov momentum
# - learning_rate: 0.01 (controls step size in gradient descent)
# - momentum: 0.9 (accelerates convergence in consistent gradient directions)
# - nesterov: True (uses Nesterov Accelerated Gradient for faster convergence)
opt_sgdm = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model
# - optimizer: SGDM with Nesterov acceleration
# - loss: categorical_crossentropy (standard for multi-class classification)
# - metrics: accuracy (for monitoring training performance)
model_sgdm.compile(optimizer=opt_sgdm, loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled successfully with SGDM + Nesterov Accelerated Gradient")
print(f"Optimizer: SGD (lr=0.01, momentum=0.9, nesterov=True)")
print(f"Loss Function: Categorical Crossentropy")
print(f"Metrics: Accuracy")
