import numpy as np
from keras import Sequential
from keras.layers import Dense

"""
Create training data and labels
"""
data = np.random.random(10000)
data.shape = 10000, 1  # observation, number of input parameter
labels = np.array(data >= .5, dtype=int)

"""
Creating a model with sequential
input layer  - 1 neuron
hidden layer - 8 densely connected neurons
output layer - 2 (0 or 1) neurons
"""
model = Sequential(
    [
        Dense(units=8, input_shape=(1,), activation='relu'),  # hidden layer
        Dense(units=2, activation='softmax')  # output layer
    ]
)

"""
Compile the model
"""
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


"""
Train the model
"""
model.fit(
    x=data,
    y=labels,
    epochs=10,
    verbose=2,  # output
    validation_split=.1  # Keep some data for validation
)

"""
Testing the model
"""
test_data = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
predictions = model.predict(test_data)
print(np.argmax(predictions, axis=1))
