from Data_processing.processing import data_collect, data_transform
from Models.naive_bayes_classifier import NaiveBayesClassifier
from Data_visualisation.visualise import classify_score
import pickle
import sys

if __name__ == '__main__':
    data_collect()
    X_train, X_test, Y_train, Y_test = data_transform()
    model = NaiveBayesClassifier()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test, train=True)
    classify_score(predicted, Y_test)
    model = pickle.load(open('Models/Saved/nb_classifier.pickle', 'rb'))
    vectorizer = pickle.load(open('Models/Vectorizer/vectorizer.pickle', 'rb'))
    while True:
        sys.stdout.write("Hi!\nSymptoms:\n")
        symptoms = input("")
        result_ar = vectorizer.transform([symptoms]).toarray()
        result = model.predict(result_ar)


