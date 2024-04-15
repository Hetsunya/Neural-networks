import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Data generation
x = np.arange(-20, 20, 0.1)
y = np.sin(x) + np.sin(np.sqrt(2)*x)

# Data split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=13)

# Calculate actual period and offset for training
t_train = X_train // (2*np.pi)
fi_train = X_train % (2*np.pi)

# Create and train subnetworks

# Network for predicting period (t)
model_period = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_period.compile(loss='mse', optimizer='adam')
model_period.fit(X_train, t_train, epochs=300, batch_size=16, verbose=1)

# Network for predicting offset (fi)
model_offset = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_offset.compile(loss='mse', optimizer='adam')
model_offset.fit(X_train, fi_train, epochs=300, batch_size=16, verbose=1)

# Combine pre-trained networks and create final model

# Input layer
input_layer = keras.Input(shape=(1,))

# Get period and offset
period = model_period(input_layer)
offset = model_offset(input_layer)

# Concatenate
combined = layers.Concatenate()([period, offset])

# Output layer
output_layer = layers.Dense(1)(combined)

# Create and evaluate complex model (combined before training)
model_complex1 = keras.Model(inputs=input_layer, outputs=output_layer)
model_complex1.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_complex1.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)
_, accuracy_complex1 = model_complex1.evaluate(X_test, y_test, verbose=0)
print("Accuracy (Combined Before Training):", accuracy_complex1)

# Create and evaluate complex model (combined after training)

# Freeze subnetwork weights
model_period.trainable = False
model_offset.trainable = False

# Create model with frozen subnetworks and trainable output layer
model_complex2 = keras.Model(inputs=input_layer, outputs=output_layer)
model_complex2.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model_complex2.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1)
_, accuracy_complex2 = model_complex2.evaluate(X_test, y_test, verbose=1)
print("Accuracy (Combined After Training):", accuracy_complex2)

# Visualize results (using the last model)
y_pred = model_complex2.predict(X_test)
plt.plot(x, y, label='f(x)')
plt.scatter(X_test, y_pred, label='Predictions', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Approximation of f(x)')
plt.show()