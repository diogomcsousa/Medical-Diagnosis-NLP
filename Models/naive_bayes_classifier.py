from sklearn.naive_bayes import MultinomialNB, GaussianNB
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from Data_visualisation.visualise import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self):
        # Build the model
        self.model = MultinomialNB()
        self.probabilities = {
            (0.10, 0.24): 'Improbable',
            (0.25, 0.39): 'Doubtfully',
            (0.40, 0.54): 'Probably',
            (0.55, 0.69): 'Likely',
            (0.70, 0.84): 'High likely',
            (0.85, 1.00): 'Almost certainly'
        }

    def get_probability(self, value):
        for key, val in self.probabilities.items():
            if key[0] <= value <= key[1]:
                return val

    def fit(self, X, Y):
        # Train the model using the training data
        params = {
            'alpha': np.linspace(0.5, 1.5, 6)
            # 'var_smoothing': np.logspace(0, -9, num=100),
        }
        self.model.fit(X, Y)
        # classifier = GridSearchCV(self.model, params, verbose=1, cv=10, n_jobs=-1, return_train_score=True)
        #
        # classifier.fit(X, Y)
        # print("Best Estimator: \n{}\n".format(classifier.best_estimator_))
        # print("Best Parameters: \n{}\n".format(classifier.best_params_))
        # print("Best Validation Score: \n{}\n".format(classifier.best_score_))
        # title = "Learning Curves (Naive Bayes)"
        #
        # plot_learning_curve(classifier.best_estimator_, title, X, Y)
        # plt.show()
        pickle.dump(self, open('Models/Saved/nb_classifier.pickle', 'wb'))

    def predict(self, X, train=False):
        # Predict the categories of the test data
        if train:
            return self.model.predict(X)

        else:
            df_1 = pd.read_csv('Data/disease.csv', sep=';', names=['disease', 'description', 'action'])
            y_ar = self.model.predict_proba(X)

            diagnosis = {}
            for e in range(len(y_ar[0])-1):
                if y_ar[0][e] >= 0.1:
                    print(y_ar[0][e])
                    diagnosis[self.model.classes_[e]] = self.get_probability(round(y_ar[0][e], 2))

            description = df_1.loc[df_1['disease'] == self.model.classes_[np.argmax(y_ar)]].values[0][1]
            action = df_1.loc[df_1['disease'] == self.model.classes_[np.argmax(y_ar)]].values[0][2]

            return self.model.classes_[np.argmax(y_ar)], diagnosis, description, action
