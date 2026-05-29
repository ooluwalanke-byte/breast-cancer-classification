# Configuration B: RMSprop
from tensorflow.keras.optimizers import RMSprop
from optimized_network import build_optimized_network

# Configuration parameters
INPUT_FEATURES = 30  # Number of input features
CLASSES = 3          # Number of output classes

# Build model
model_rmsprop = build_optimized_network(INPUT_FEATURES, CLASSES)

# Initialize RMSprop optimizer
# - learning_rate: 0.001 (controls step size in gradient descent)
# - rho: 0.9 (decay rate for moving average of squared gradients)
# - epsilon: 1e-07 (small constant for numerical stability to prevent division by zero)
opt_rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)

# Compile the model
# - optimizer: RMSprop (adaptive learning rate optimization)
# - loss: categorical_crossentropy (standard for multi-class classification)
# - metrics: accuracy (for monitoring training performance)
model_rmsprop.compile(optimizer=opt_rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled successfully with RMSprop optimizer")
print(f"Optimizer: RMSprop (lr=0.001, rho=0.9, epsilon=1e-07)")
print(f"Loss Function: Categorical Crossentropy")
print(f"Metrics: Accuracy")
