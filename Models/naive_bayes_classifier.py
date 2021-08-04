from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        # Build the model
        self.model = MultinomialNB()

    def fit(self, X, Y):
        # Train the model using the training data
        self.model.fit(X, Y)
        pickle.dump(self, open('Models/Saved/nb_classifier.pickle', 'wb'))

    def predict(self, X, train=False):
        # Predict the categories of the test data
        if train:
            return self.model.predict(X)

        else:
            df_1 = pd.read_csv('Data/disease.csv', sep=';', names=['disease', 'description', 'action'])
            print(df_1.head())
            y_ar = self.model.predict_proba(X)

            print(y_ar)
            diagnosis = []
            for e in range(len(y_ar[0])-1):
                if y_ar[0][e] >= 0.1:
                    diagnosis.append(self.model.classes_[e])

            print(diagnosis)
            print(self.model.classes_[np.argmax(y_ar)])
            print(df_1.loc[df_1['disease'] == self.model.classes_[np.argmax(y_ar)]].values[0][1])
            print(df_1.loc[df_1['disease'] == self.model.classes_[np.argmax(y_ar)]].values[0][2])

            return self.model.classes_[np.argmax(y_ar)]
