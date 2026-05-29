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
