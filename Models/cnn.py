import pandas as pd
from keras import layers
from keras.models import Sequential
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np


class CNN:
    def __init__(self):
        # Build the model
        self.model = Sequential()

    def fit(self, X, Y):
        # Train the model using the training data
        self.model.add(layers.Dense(10, input_dim=len(X[0]), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(90, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())
        y = pd.DataFrame(Y)
        y_train = pd.get_dummies(y).values

        history = self.model.fit(X, y_train, validation_split=0.2, epochs=10, verbose=1, batch_size=128)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        self.model.save('Models/Saved/cnn_classifier.h5')

    def predict(self, X, Y, train=False):
        # Predict the categories of the test data
        y = pd.DataFrame(Y)
        y_test = pd.get_dummies(y).values
        y_test_final = pd.DataFrame(data=y_test).idxmax(axis=1).values
        y_pred = np.argmax(self.model.predict(X), axis=-1)

        accuracy = accuracy_score(y_test_final, y_pred)
        f1 = f1_score(y_test_final, y_pred, average='macro')
        recall = recall_score(y_test_final, y_pred, average='macro')
        precision = precision_score(y_test_final, y_pred, average='macro')

        print("score_test: %.4f" % accuracy)
        print("f1_test: %.4f" % f1)
        print("recall_test: %.4f" % recall)
        print("precision_test: %.4f" % precision)
