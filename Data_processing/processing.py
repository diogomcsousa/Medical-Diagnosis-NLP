import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download("wordnet")
nltk.download("punkt")

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def normalise(sentence):
    """
    Perform normalisation of a sentence with stop word removal, tokenization and lemmatization
    """
    symptoms = []
    for word, tag in pos_tag(word_tokenize(sentence.lower())):
        if word not in STOP_WORDS:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                symptoms.append(word)
            else:
                symptoms.append(LEMMATIZER.lemmatize(word, wntag))
    return symptoms


def data_collect():
    """
    Create final data set based on initial data sets
    """

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
        dermatology = ['', 'psoriasis', 'seborrheic dermatitis',
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

    final_dataset = pd.DataFrame(columns=['symptoms', 'diagnosis'])
    for tup in new_dataset.itertuples():
        symptoms = normalise(tup[1])
        final_dataset = final_dataset.append({'symptoms': ' '.join(symptoms), 'diagnosis': tup[2]},
                                             ignore_index=True)
    final_dataset.to_csv("Data/medical_diagnosis.csv", index=False)


def data_transform():
    """
    Vectorize final data set to be used in model training
    """

    corpus = pd.read_csv('Data/medical_diagnosis.csv')['symptoms']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    pickle.dump(vectorizer, open('Models/Vectorizer/vectorizer.pickle', 'wb'))
    y = pd.read_csv('Data/medical_diagnosis.csv')['diagnosis']
    diseases = set(y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, Y_train, Y_test, diseases
