import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# ===========================================================
#                          Load Data
# ===========================================================
# TODO: Load data here.
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y>0).astype(int).reshape(y.shape[0])

# ===========================================================
#                          Load Data
# ===========================================================


# ===========================================================
#                       Feature ranking
# =========================================================== 
# TODO: Design your score function for feature selection
# TODO: To use the provided evaluation sample code, you need to generate ranking_idx, which is the sorted index of feature
# ===========================================================
#                       Feature ranking
# =========================================================== 


# ===========================================================
#                       Feature evaluation
# =========================================================== 
# Use a simple dicision tree with 5-fold validation to evaluate the feature selection result.
# You can try other classifier and hyperparameter.
score_history = []
for m in range(5, 2001, 5):
    # Select Top m feature
    x_subset = x[:, ranking_idx[:m]]

    # Build random forest
    clf = DecisionTreeClassifier(random_state=0)
    # clf = SVC(kernel='rbf', random_state=0) #build SVM

    # Calculate validation score
    scores = cross_val_score(clf, x_subset, y, cv=5)

    # Save the score calculated with m feature
    score_history.append(scores.mean())

# Report best accuracy.
print(f"Max of Decision Tree: {max(score_history)}")
print(f"Number of features: {np.argmax(score_history)*5+5}")
# ===========================================================
#                       Feature evaluation
# =========================================================== 

# ===========================================================
#                       Visualization
# =========================================================== 
plt.plot(range(5, 2001, 5), score_history, c='blue')
plt.title('Original')
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.legend(['Decision Tree'])
plt.savefig('1-3_result.png')
# ===========================================================
#                       Visualization
# =========================================================== 