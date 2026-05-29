# Configuration C: Adam Optimizer
from tensorflow.keras.optimizers import Adam
from optimized_network import build_optimized_network

# Configuration parameters
INPUT_FEATURES = 30  # Number of input features
CLASSES = 3          # Number of output classes

# Build model
model_adam = build_optimized_network(INPUT_FEATURES, CLASSES)

# Initialize Adam optimizer
# - learning_rate: 0.001 (controls step size in gradient descent)
# - beta_1: 0.9 (exponential decay rate for first moment estimates)
# - beta_2: 0.999 (exponential decay rate for second moment estimates)
# - epsilon: 1e-07 (small constant for numerical stability to prevent division by zero)
opt_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# Compile the model
# - optimizer: Adam (Adaptive Moment Estimation, combines momentum and RMSprop)
# - loss: categorical_crossentropy (standard for multi-class classification)
# - metrics: accuracy (for monitoring training performance)
model_adam.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled successfully with Adam optimizer")
print(f"Optimizer: Adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)")
print(f"Loss Function: Categorical Crossentropy")
print(f"Metrics: Accuracy")
