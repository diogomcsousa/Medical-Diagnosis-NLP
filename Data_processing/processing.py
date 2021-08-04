import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


def data_collect():
    df_1 = pd.read_csv('Data/training.csv')
    df_2 = pd.read_csv('Data/diagnosis.csv')
    df_3 = pd.read_csv('Data/dermatology.csv')
    df_4 = pd.read_csv('Data/medical_data.csv')

    new_dataset = pd.DataFrame(columns=['symptoms', 'diagnosis'])
    new_dataset = new_dataset.append(df_4, ignore_index=True)

    for tup in df_1.itertuples(index=False):
        symptoms = []
        diagnosis = tup[len(tup) - 1].lower()
        for e in range(len(tup)):
            if tup[e] == 1:
                symptoms.append(df_1.columns[e].replace('_', ' ').lower())
        new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': diagnosis}, ignore_index=True)

    for tup in df_2.itertuples(index=False):
        symptoms = [tup[0].replace(',', '.')]
        for e in range(1, len(tup) - 2):
            if tup[e] == "yes":
                symptoms.append(df_2.columns[e].replace(',', '').lower())
        if tup[len(tup) - 1] == "yes":
            new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': "nephritis"},
                                             ignore_index=True)
        if tup[len(tup) - 2] == "yes":
            new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms),
                                              'diagnosis': "urinary tract infection"},
                                             ignore_index=True)

    for tup in df_3.itertuples(index=False):
        dermatology = ['', 'psoriasis', 'seboreic dermatitis',
                       'lichen planus', 'pityriasis rosea',
                       'chronic dermatitis', 'pityriasis rubra pilaris']

        if tup[len(tup) - 2] != '?':
            symptoms = [str(tup[len(tup) - 2])]
        else:
            symptoms = []
        for e in range(len(tup) - 2):
            if tup[e] != 0 and tup[e] != '?':
                symptoms.append(df_3.columns[e].replace(',', '').lower())
        new_dataset = new_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': dermatology[tup[len(tup) - 1]]},
                                         ignore_index=True)

    new_dataset.to_csv("Data/medical_diagnosis.csv", index=False)


def data_transform():
    corpus = pd.read_csv('Data/medical_diagnosis.csv')['symptoms']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    pickle.dump(vectorizer, open('Models/Vectorizer/vectorizer.pickle', 'wb'))
    y = pd.read_csv('Data/medical_diagnosis.csv')['diagnosis']
    diseases = set(y)
    print(sorted(diseases))
    print(len(diseases))

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, Y_train, Y_test