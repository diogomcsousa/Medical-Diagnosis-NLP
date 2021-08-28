from Data_processing.processing import data_collect, data_transform, normalise
from Models.naive_bayes_classifier import NaiveBayesClassifier
from Data_visualisation.visualise import classify_score
import pickle
import sys


def loop(symp):
    symp += " " + input("")
    sentence = normalise(symp)
    processed_sentence = vectorizer.transform([' '.join(sentence)]).toarray()
    result, diagnosis, description, action = model.predict(processed_sentence)
    sys.stdout.write("Thank you for your information.\n")
    sys.stdout.write("Based on your symptoms the diagnosis could be:\n\n")
    for key, val in diagnosis.items():
        sys.stdout.write(f"\t- {val} that you have {key.capitalize()}\n\n")

    sys.stdout.write(f"{description}\n\n")
    sys.stdout.write("Recommended action:\n")
    sys.stdout.write(f"{action}\n")
    sys.stdout.write("Please contact your doctor to obtain more information.\n\n")
    sys.stdout.write("Do you have more symptoms to include?(yes/no)\n\n")
    new_answer = input("")

    return symp, new_answer


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        data_collect()
        X_train, X_test, Y_train, Y_test, labels = data_transform()
        model = NaiveBayesClassifier()
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test, train=True)
        classify_score(predicted, Y_test)
    model = pickle.load(open('Models/Saved/nb_classifier.pickle', 'rb'))
    vectorizer = pickle.load(open('Models/Vectorizer/vectorizer.pickle', 'rb'))
    sys.stdout.write("Hello, could you please describe your symptoms?\n\n")
    symptoms, answer = loop("")
    while True:
        if answer.lower() == "yes":
            sys.stdout.write("Please add new symptoms.\n\n")
            symptoms, answer = loop(symptoms)
        else:
            sys.stdout.write("Thank you, have a good day.\n")
            break


