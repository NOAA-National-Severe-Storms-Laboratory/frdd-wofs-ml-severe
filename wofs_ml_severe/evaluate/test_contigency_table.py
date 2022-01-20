
from wofs_ml.evaluate.metrics import Metrics

import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import precision_recall_curve

X, y = datasets.make_classification(n_samples=1000, n_features=3,
                                    n_informative=2, n_redundant=0,
                                    class_sep = 1.0,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)


clf = RandomForestClassifier(n_jobs=9, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

#sr, pod, _ = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1]) 

pod, pofd, sr = Metrics.performance_curve(y_test, clf.predict_proba(X_test)[:,1])

print(pod, sr)


