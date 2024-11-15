
import time
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
RANDOM_SEED = 7
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, precision_recall_fscore_support

# train, test and save model
def exp_clf_with_feature_selected(clf_model, X_train, X_test, y_train, y_test,verbose=True):
    start = time.time()

    clf_model.fit(X_train, y_train)
    predictions = clf_model.predict(X_test.values)

    # Calculate metrics
    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1-score': report['macro avg']['f1-score']
    }
    for class_name in report.keys():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[class_name+'_precision'] = report[class_name]['precision']
            metrics[class_name+'_recall'] = report[class_name]['recall'],
            metrics[class_name+'_f1-score'] = report[class_name]['f1-score']

    feature_columns = list(X_train.columns)
    num_classes = y_train.nunique()
    class_names = list(y_train.unique())

    model_filename = f"./models/{clf_model.__class__.__name__}_model"
    model_filename += f"_{num_classes}cls_{len(feature_columns)}feat_{round(report['accuracy']*100)}acc.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(clf_model, file)

    results = {**metrics,
        'num_classes': num_classes,
        'class_names': class_names,
        'model_filename': model_filename,
        'feature_columns': feature_columns,
    }

    if verbose:
        print(f"Model Name: {clf_model.__class__.__name__};\nTrain set shape {X_train.shape}, num of class {num_classes}")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        probabilities = clf_model.predict_proba(X_test.values)
        print('Probabilities distribution:\n', pd.DataFrame(probabilities, columns=clf_model.classes_).describe())
    print(f"Model: {clf_model.__class__.__name__};Time taken: {round(time.time()-start, 3)} seconds.\n")

    return results, clf_model

# test trained model with test set 
def test_clf_model(clf_model, X_test, y_test,verbose=True):
    start = time.time()

    predictions = clf_model.predict(X_test.values)

    # Calculate metrics
    report = classification_report(y_test, predictions, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1-score': report['macro avg']['f1-score']
    }
    for class_name in report.keys():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[class_name+'_precision'] = report[class_name]['precision']
            metrics[class_name+'_recall'] = report[class_name]['recall'],
            metrics[class_name+'_f1-score'] = report[class_name]['f1-score']

    num_classes = y_test.nunique()
    results = {**metrics,
        'num_classes': num_classes,
    }

    if verbose:
        print(f"Model Name: {clf_model.__class__.__name__};\nTest set shape {X_test.shape}, num of class {num_classes}")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        probabilities = clf_model.predict_proba(X_test.values)
        print('Probabilities distribution:\n', pd.DataFrame(probabilities, columns=clf_model.classes_).describe())
    print(f"Model: {clf_model.__class__.__name__};Time taken: {round(time.time()-start, 3)} seconds.\n")

    return results, clf_model