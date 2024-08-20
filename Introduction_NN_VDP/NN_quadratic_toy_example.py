# Neural Network that predicts y=x^2 in interval [-1,1]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Custom function for data splitting
def custom_train_validation_split(data, labels, validation_ratio=0.2):
    num_samples = len(data)
    num_validation_samples = int(num_samples * validation_ratio)

    # Randomly shuffle the data and labels
    indices = np.arange(num_samples).astype(int)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split the data and labels
    X_val = data[:num_validation_samples]
    y_val = labels[:num_validation_samples]
    X_train = data[num_validation_samples:]
    y_train = labels[num_validation_samples:]

    return X_train, X_val, y_train, y_val


def square_func(x: np.ndarray):
    y = np.square(x)
    return y
# Step 1: Generate Data
x = np.random.rand(1000, 1)*2 -1 #[-1,1]
y = square_func(x)
# y = x + 0.1 * np.random.randn(1000, 1) # to make it random

# Step 3: Build the Model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu')) # 1 input dimension
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

# Step 4: Compile the Model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01)) # default optimizer Adam, learning rate 0.01 common starting point

# Step 5: Split Data into Training and Validation Sets using the custom function
X_train, X_val, y_train, y_val = custom_train_validation_split(x, y, validation_ratio=0.2)

# Step 6: Train the Model with Validation Data
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Step 7: Plot the Loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# log
plt.figure()
plt.semilogy(epochs, train_loss, 'bo', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss (Logarithmic Scale)')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.legend()

plt.show()

# Step 8: Test the Model With Own Values
x_val = np.linspace(-1.5,2.5,40)
x_dat = np.array(x_val).reshape(-1,1)
y_dat = square_func(x_dat)

predictions = model.predict(x_dat)
# Step 8.1 plot differences
plt.figure()
plt.scatter(x_dat, y_dat, s=3, alpha=0.5, label='Validation coord')
plt.scatter(x_dat, predictions, s=5, alpha=0.5, label='Prediction coord')
plt.plot([-1,-1],[0,1], linestyle ='dashed', color='black')
plt.plot([-1,1],[1,1], linestyle ='dashed', color='black')
plt.plot([1,1],[1,0], linestyle ='dashed', color='black')
plt.plot([1,-1],[0,0], linestyle ='dashed', color='black')
plt.title('Validation and Prediction of chosen values as coordinates for y=x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()

# Conclusion: works very good in the trained area of 0-1, not so much outside of that