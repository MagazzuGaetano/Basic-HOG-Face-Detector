import seaborn as sns
import matplotlib.pyplot as plt

from numpy import loadtxt
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib


def show_cm(y_pred, y_test):
    mat = confusion_matrix(y_pred, y_test)
    sns.heatmap(
        mat.T,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("score = {}".format(accuracy_score(y_test, y_pred)))
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.show()


classes = ["head", "not head"]

# load data
X_test = loadtxt("X_test.csv")
y_test = loadtxt("y_test.csv")

print(X_test.shape)

# load model
_, _, model  = joblib.load("model.sav")

# predict
y_pred = model.predict(X_test)

# confusion matrix
show_cm(y_pred, y_test)
