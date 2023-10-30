from sklearn import svm
import pandas as pd
import numpy as np

from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pylab as plt

def specificity_loss_func(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return tn/(tn+fp)


def print_summary(results):
    print("Accuracy: %.2f%% ± %.2f%%" %
          (np.mean(results['acc']) * 100, np.std(results['acc']) * 100))
    print("Sensitivity: %.2f%% ± %.2f%%" %
          (np.mean(results['sens']) * 100, np.std(results['sens']) * 100))
    print("Specificity: %.2f%% ± %.2f%%" %
          (np.mean(results['spec']) * 100, np.std(results['spec']) * 100))
    print("F1-score: %.2f%% ± %.2f%%" %
          (np.mean(results['f1_score']) * 100, np.std(results['f1_score']) * 100))
    print("AUC: %.2f ± %.2f" %
          (np.mean(results['auc']) * 100, np.std(results['auc']) * 100))


def get_model(kernel='rbf', C=1.0, gamma='scale', probability=False):
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma,
                    probability=probability, class_weight='balanced')
    return model


def read_data(path):
    data = pd.read_csv(path, usecols=lambda column: column not in ["class"])
    data = np.array(data)

    classes = pd.read_csv(path, usecols=["class"])
    Y = classes.values.ravel()

    # Minmax scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(data)

    return X, Y


def train_and_valid(model, X, Y):
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    # Results dict
    results = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}

    # Needed for ROC
    fig1 = 0

    # Cross-validation
    for train, test in cv.split(X, Y):
        model = model.fit(X[train], Y[train])

        # Generating predictions
        predicted = model.predict(X[test])
        predicted_proba = model.predict_proba(X[test])

        results['acc'].append(accuracy_score(Y[test], predicted))
        results['f1_score'].append(f1_score(Y[test], predicted))
        results['spec'].append(specificity_loss_func(Y[test], predicted))
        results['sens'].append(recall_score(Y[test], predicted))
        results['auc'].append(roc_auc_score(Y[test], predicted_proba[:, 1]))

    

    return results


if __name__ == "__main__":
    model = get_model(probability=True)
    X, Y = read_data("data/radiomics.csv")
    # X, Y = read_data("data/deep_radiomics.csv")
    results = train_and_valid(model, X, Y)
    print_summary(results)
