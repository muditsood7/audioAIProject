import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

# input array: array([[0.1, 0.2], [0.2, 0.2]])
# output array: array([0.3], [0.4]])

# generating training and test sets. test set is 30% of the total set.
def generate_dataset(num_samples, test_size):
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    # print("x_test: \n {}".format(x_test))
    # print("y_test: \n {}".format(y_test))

    # build model: 2 neurons --> 5 neurons --> 1 neuron
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # train model
    model.fit(x_train, y_train, epochs=100)

    # evaluate model
    print("Model Evaluation: \n")
    model.evaluate(x_test, y_test, verbose=1)

    # make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2], [0.5, 0.4]])
    predictions = model.predict(data)

    print("Some predictions: ")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

