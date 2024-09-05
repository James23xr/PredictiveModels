import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

# TensorFlow model
def create_tf_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self, input_shape):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Function to train and evaluate models
def train_and_evaluate(X, y, epochs=100, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TensorFlow
    tf_model = create_tf_model(X.shape[1])
    
    # Custom training loop with tqdm for TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    tf_progress_bar = tqdm(range(epochs), desc="TensorFlow Training")
    for epoch in tf_progress_bar:
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred = tf_model(x_batch, training=True)
                loss = mse_loss(y_batch, y_pred)
            grads = tape.gradient(loss, tf_model.trainable_variables)
            tf_model.optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))
        tf_progress_bar.set_postfix({"loss": float(loss)})
    
    tf_pred = tf_model.predict(X_test)
    tf_mse = mean_squared_error(y_test, tf_pred)
    
    # PyTorch
    torch_model = PyTorchModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(torch_model.parameters())
    
    torch_progress_bar = tqdm(range(epochs), desc="PyTorch Training")
    for epoch in torch_progress_bar:
        optimizer.zero_grad()
        outputs = torch_model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train).unsqueeze(1))
        loss.backward()
        optimizer.step()
        torch_progress_bar.set_postfix({"loss": loss.item()})
    
    torch_model.eval()
    with torch.no_grad():
        torch_pred = torch_model(torch.FloatTensor(X_test))
    torch_mse = mean_squared_error(y_test, torch_pred.numpy())
    
    print(f"TensorFlow MSE: {tf_mse}")
    print(f"PyTorch MSE: {torch_mse}")
    
    return tf_mse, torch_mse

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Train and evaluate models
tf_mse, torch_mse = train_and_evaluate(X, y)

# Calculate improvement
improvement = ((max(tf_mse, torch_mse) - min(tf_mse, torch_mse)) / max(tf_mse, torch_mse)) * 100
print(f"Model accuracy improved by {improvement:.2f}%")
