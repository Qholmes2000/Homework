# test.py
import numpy as np
from utils import load_model
from model import NeuralNetwork

def evaluate_test_set(model_config, model_path, X_test, y_test):
    # Reconstruct model architecture
    model = NeuralNetwork(
        layer_sizes=[model_config['input_dim'], 
                    model_config['hidden_size'], 
                    model_config['hidden_size']//2, 
                    model_config['num_classes']],
        activations=[model_config['activation'], model_config['activation']],
        init_methods=[model_config['init_method']]*3,
        reg_strength=model_config['reg_strength']
    )
    
    # Load weights
    load_model(model, model_path)
    
    # Predict
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_true)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    return accuracy