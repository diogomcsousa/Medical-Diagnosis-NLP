import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def convert():
    df = pd.read_csv('training.csv')
    df1 = pd.read_csv('diagnosis.csv')
    df2 = pd.read_csv('dermatology.csv')

    new_dataset = pd.DataFrame(columns=['symptoms', 'diagnosis'])

    for tup in df.itertuples(index=False):
        symptoms = []
        diagnosis = tup[len(tup)-1].lower()
        for e in range(len(tup)):
            if tup[e] == 1:
                symptoms.append(df.columns[e].replace('_', ' ').lower())
        new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': diagnosis}, ignore_index=True)

    for tup in df1.itertuples(index=False):
        symptoms = [tup[0].replace(',', '.')]
        for e in range(1, len(tup) - 2):
            if tup[e] == "yes":
                symptoms.append(df1.columns[e].replace(',', '').lower())
        if tup[len(tup)-1] == "yes":
            new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': "nephritis"},
                                             ignore_index=True)
        if tup[len(tup)-2] == "yes":
            new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms),
                                              'diagnosis': "urinary tract infection"},
                                             ignore_index=True)

    for tup in df2.itertuples(index=False):
        dermatology = ['', 'psoriasis', 'seboreic dermatitis',
                       'lichen planus', 'pityriasis rosea',
                       'cronic dermatitis', 'pityriasis rubra pilaris']

        if tup[len(tup) - 2] != '?':
            symptoms = [str(tup[len(tup)-2])]

        for e in range(len(tup) - 2):
            if tup[e] != 0 and tup[e] != '?':
                symptoms.append(df2.columns[e].replace(',', '').lower())
        new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': dermatology[tup[len(tup)-1]]},
                                         ignore_index=True)

    new_dataset.to_csv("medical_diagnosis.csv", index=False)
    corpus = new_dataset['symptoms']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    y = new_dataset['diagnosis']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4)

    # Build the model
    model = MultinomialNB()
    # Train the model using the training data
    model.fit(X_train, Y_train)
    # Predict the categories of the test data
    predicted_categories = model.predict(X_test)

    mat = confusion_matrix(Y_test, predicted_categories)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, cmap=plt.cm.get_cmap('Blues', 6))
    fig.colorbar(cax)
    #sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=Y_train,
         #       yticklabels=Y_train)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    print(vectorizer.inverse_transform(X_test[:2]))
    print(predicted_categories[:2])
    print(f1_score(Y_test, predicted_categories, average='macro'))
    print(model.classes_)

    test = ['burning urethra micturition pain 41.0']
    test_ar = vectorizer.transform(test).toarray()
    print(test_ar.shape)
    y_ar = model.predict_proba(test_ar)
    print(model.classes_[np.argmax(y_ar)])
    print(y_ar)


if __name__ == '__main__':
    convert()
