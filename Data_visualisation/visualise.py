from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt


def classify_score(predicted, true):
    mat = confusion_matrix(true, predicted)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, cmap=plt.cm.get_cmap('Blues', 6))
    fig.colorbar(cax)
    # sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=Y_train,
    #       yticklabels=Y_train)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    print(f1_score(true, predicted, average='macro'))