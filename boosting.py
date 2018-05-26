import numpy as np

from collections import Counter

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

seed = 104

x, y = make_classification(n_samples=1000,n_features=20,n_informative=8,n_redundant=3,n_repeated=2,random_state=seed)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=seed)

print("Train label distribution:")
print(Counter(y_train))
print("\nTest label distribution:")
print(Counter(y_test))

decision_tree = DecisionTreeClassifier(random_state=seed)

# train classifier
decision_tree.fit(x_train, y_train)

# predict output
decision_tree_y_pred = decision_tree.predict(x_test)
decision_tree_y_pred_prob = decision_tree.predict_proba(x_test)

# evaluation
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
decision_tree_logloss = log_loss(y_test, decision_tree_y_pred_prob)

print("== Decision Tree ==")
print("Accuracy: {0:.2f}".format(decision_tree_accuracy))
print("Log loss: {0:.2f}".format(decision_tree_logloss))
print("Number of nodes created: {}".format(decision_tree.tree_.node_count))

print("True labels:")
print(y_test[:5,])
print('\nPredicted labels:')
print(decision_tree_y_pred[:5,])
print('\nPredicted probabilities:')
print(decision_tree_y_pred_prob[:5,])


# adaboost
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    algorithm='SAMME',
    n_estimators=1000,
    random_state=seed)

adaboost.fit(x_train,y_train)
adaboost_y_pred = adaboost.predict(x_test)
adaboost_y_pred_prob = adaboost.predict_proba(x_test)

adaboost_accuracy = accuracy_score(y_test,adaboost_y_pred)
adaboost_logloss = log_loss(y_test,adaboost_y_pred_prob)
print("== AdaBoost ==")
print("Accuracy: {0:.2f}".format(adaboost_accuracy))
print("Log loss: {0:.2f}".format(adaboost_logloss))

print("True labels:")
print(y_test[:5,])
print('\nPredicted labels:')
print(adaboost_y_pred[:5,])
print('\nPredicted probabilities:')
print(adaboost_y_pred_prob[:5,])
print("Error: {0:.2f}".format(adaboost.estimator_errors_[0]))
print("Tree importance: {0:.2f}".format(adaboost.estimator_weights_[0]))

# GradientBoosting Trees
gbc = GradientBoostingClassifier(
    max_depth=1,
    n_estimators=1000,
    warm_start=True,
    random_state=seed
)

gbc.fit(x_train,y_train)

# predictions
gbc_y_pred = gbc.predict(x_test)
gbc_y_pred_prob = gbc.predict_proba(x_test)

# log loss
gbc_accuracy = accuracy_score(y_test,gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_pred_prob)

print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(gbc_accuracy))
print("Log loss: {0:.2f}".format(gbc_logloss))

print("True labels:")
print(y_test[:5,])
print('\nPredicted labels:')
print(gbc_y_pred[:5,])
print('\nPredicted probabilities:')
print(gbc_y_pred_prob[:5,])


def main():
    return
main()