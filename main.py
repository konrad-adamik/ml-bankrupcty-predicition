import pandas as pd

import matplotlib.pyplot as plt
import itertools

import graphviz

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

# Load data
data = pd.read_csv("data/data.csv")

# Divide data into input features and target (output) features
dataset = data.values[:, 1:]
target = data.values[:, 0]

# Split data into training and testing datasets
training_data, testing_data, training_target, testing_target = \
    train_test_split(dataset, target.reshape(-1, 1), test_size=0.2, random_state=1)

# Create and fit the decision tree
decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
decision_tree = decision_tree.fit(training_data, training_target)

# Evaluation metrics for decision tree

# Plain accuracy
print("Accuracy score:")
print(accuracy_score(testing_target, decision_tree.predict(testing_data)))

# Visualize confusion matrix
cf_matrix = confusion_matrix(testing_target, decision_tree.predict(testing_data))

print(cf_matrix)

plot1 = plt.figure(figsize=(10, 7))
plt.imshow(cf_matrix, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.ylabel("Original Label")
plt.xlabel("Predicted Label")

plt.xticks([0, 1], ["Not bankrupt", "Bankrupt"])
plt.yticks([0, 1], ["Not bankrupt", "Bankrupt"])

thresh = cf_matrix.max() / 2.0
for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
    plt.text(j, i, cf_matrix[i, j], horizontalalignment="center",
             color="white" if cf_matrix[i, j] > thresh else "black")

plt.colorbar()

# Visualize ROC

y_train_score = decision_tree.predict_proba(training_data)[:, 1]
y_test_score = decision_tree.predict_proba(testing_data)[:, 1]
auc_train = roc_auc_score(training_target, y_train_score)
auc_test = roc_auc_score(testing_target, y_test_score)

fpr, tpr, _ = roc_curve(testing_target, y_test_score)
roc_auc = auc(fpr, tpr)
plot2 = plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label="ROC curve(area= % 0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend(loc="lower right")

plt.show()

# Visualize decision tree into the file
dot_data = tree.export_graphviz(decision_tree, feature_names=data.columns[1:],
                                class_names=["Not bankrupt", "Bankrupt"], out_file=None, filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render("result/decision_tree")
