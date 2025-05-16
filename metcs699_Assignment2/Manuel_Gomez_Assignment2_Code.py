import pandas as pd
import math
from scipy.spatial.distance import hamming, euclidean, cityblock
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt


# # Problem 1 - Calculate Distance (Categorical Variables)
#
# P4 = ["management", "married", "tertiary", "No", "yes", "yes", "unknown"]
# P5 = ["blue-collar", "single", "secondary", "No", "yes", "no", "unknown"]
# P9 = ["entrepreneur", "married", "tertiary", "No", "yes", "no", "unknown"]
#
# # def hamming_distance(x, y):
# #     # sum of all mismatches between two observations divided by total variables
# #     return sum(xi != yi for xi, yi in zip(x, y)) / len(x)
#
# # distance_P4_P5 = hamming_distance(P4, P5)
# # distance_P4_P9 = hamming_distance(P4, P9)
#
# distance_P4_P5 = hamming(P4, P5)  # SciPy's hamming function
# distance_P4_P9 = hamming(P4, P9)
# print(f"Distance between P4 and P5: {distance_P4_P5}")
# print(f"Distance between P4 and P9: {distance_P4_P9}")
# # LECTURE QUESTION - IS HAMMING DISTANCE THE CORRECT CALCULATION TO USE?
#
#
#
# # Problem 2 - Calculate Distances (Manhattan & Euclidean)
# O1 = [88, 47, 32, 6]
# O2 = [97, 63, 18, 4]
# manhattan_distance = cityblock(O1, O2)  # called "cityblock" distance in SciPy
# euclidean_distance = euclidean(O1, O2)
# print(f"Manhattan distance between O1 and O2: {manhattan_distance}")
# print(f"Euclidean distance between O1 and O2: {euclidean_distance}")
#
#
#
# # Problem 3 - Data Partitioning, Optimal k-NN Model
# df = pd.read_csv('accidents1000.csv')
# X = df.drop('MAX_SEV', axis=1)  # axis 1 specifies we're dropping a column, not a row
# y = df['MAX_SEV']
#
# print("\nPercentage distribution:")
# print(y.value_counts(normalize=True))  # "fatal" is a very small minority class
#
# # 3.1 - Generate training and holdout partitions (holding out 1/3)
# X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=1/3, stratify=y, random_state=42)
#
# # 3.2 - Scale/center, fit K-NN, make predictions, confusion matrix, accuracy
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_holdout_scaled = scaler.transform(X_holdout)
#
# param_grid = {'n_neighbors': range(1, 31)}
#
# grid_search = GridSearchCV(
#     KNeighborsClassifier(),
#     param_grid,
#     cv=5,
#     scoring='accuracy'
# )
#
# grid_search.fit(X_train_scaled, y_train)
#
# # Use best k from grid search
# best_k = grid_search.best_params_['n_neighbors']
# print(f"\nBest k (from CV): {best_k}")
# print(f"Cross-validated training accuracy: {grid_search.best_score_}")
#
# final_model = KNeighborsClassifier(n_neighbors=best_k)
# final_model.fit(X_train_scaled, y_train)
# preds = final_model.predict(X_holdout_scaled)
#
# print("\nPercentage distribution (holdout):")
# print(y_holdout.value_counts())
# print(y_holdout.value_counts(normalize=True))  # "fatal" is a very small minority class
#
# cm = confusion_matrix(y_holdout, preds)
# acc = accuracy_score(y_holdout, preds)
#
# print(f"Best k: {best_k}")
# print(f"Accuracy: {acc:.4f}")
#
# # Confusion matrix visualization
# # ROWS represent the actual/true classes (what really happened), sum is the total count of actual
# # COLUMNS represent the predicted classes (what the model thought happened), sum is the total count of predictions
#
#
# # each cell is the number of in correct
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, annot_kws={'size': 16}, fmt='d', cmap='Blues',
#             xticklabels=sorted(y.unique()),
#             yticklabels=sorted(y.unique()))
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()




# # Fit k-NN model, data centered and scaled, optimal k
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),  # centers and scales data
#     ('knn', KNeighborsClassifier())
# ])
#
# # Grid search parameters
# param_grid = {
#     'knn__n_neighbors': range(1, 21)
# }
#
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
#
# best_k = grid_search.best_params_['knn__n_neighbors']
# best_model = grid_search.best_estimator_

# print(f"Best k value: {best_k}")
# print(f"Cross-validation accuracy with k={best_k}: {grid_search.best_score_:.4f}")

# Stopped here. Should study python ML concepts as I continue

#
# # Problem 4 -
# # 4.1
# df = pd.read_csv('accidents1000.csv')
# print("total rows:", len(df))
# print("unique values:", df['MAX_SEV'].unique())
#
# df = df[df['MAX_SEV'] != 'no-injury']
#
# print("total rows:", len(df))
# print("unique values:", df['MAX_SEV'].unique())
#
# X = df.drop('MAX_SEV', axis=1)  # axis 1 specifies we're dropping a column, not a row
# y = df['MAX_SEV']
#
# # 4.2 - Generate training and holdout partitions (holding out 1/3)
# X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=1/3, stratify=y, random_state=42)
#
# # 4.3 - Fit logistic regression model, make predictions, generate confusion matrix, compute accuracy and F-score
# model = LogisticRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_holdout)
#
# print("\nPercentage distribution (holdout):")
# print(y_holdout.value_counts())
# print(y_holdout.value_counts(normalize=True))  # "fatal" is a very small minority class
#
# cm = confusion_matrix(y_holdout, y_pred)
# acc = accuracy_score(y_holdout, y_pred)
# f1 = f1_score(y_holdout, y_pred, pos_label='fatal')
#
# print(f"Accuracy: {acc:.4f}")
# print(f"F1 Score: {f1:.4f}")
#
# # Confusion matrix visualization
# # ROWS represent the actual/true classes (what really happened), sum is the total count of actual FOR that entire row category
# # COLUMNS represent the predicted classes (what the model thought happened), sum is the total count of predictions FOR that entire column category
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, annot_kws={'size': 16}, fmt='d', cmap='Blues',
#             xticklabels=sorted(y.unique()),
#             yticklabels=sorted(y.unique()))
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
# # 4.4 (Answered in text document)
# # 4.5 (Answered in text document)
# # 4.6 - Apply oversampling (Random Oversampling)
#
# ros = RandomOverSampler(random_state=42)  # Note: Random over-sampling is more flexible for smaller datasets versus SMOTE
# X_train, y_train = ros.fit_resample(X_train, y_train)
#
# # print("\nPercentage distribution:")
# # print(y_train.value_counts(normalize=True))  # "fatal" is now 50% of the observations
#
# # 4.7 - Fit Logistic Regression, Predictions, Confusion Matrix, Accuracy, F-score
# model = LogisticRegression(max_iter=500) # allowing more iterations as dataset has grown
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_holdout)
#
# cm = confusion_matrix(y_holdout, y_pred)
# acc = accuracy_score(y_holdout, y_pred)
# f1 = f1_score(y_holdout, y_pred, pos_label='fatal')
#
# print(f"Accuracy (Oversampling): {acc:.4f}")
# print(f"F1 Score (Oversampling): {f1: .4f}")
#
# # Confusion matrix visualization
# # ROWS represent the actual/true classes (what really happened), sum is the total count of actual FOR that entire row category
# # COLUMNS represent the predicted classes (what the model thought happened), sum is the total count of predictions FOR that entire column category
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, annot_kws={'size': 16}, fmt='d', cmap='Blues',
#             xticklabels=sorted(y.unique()),
#             yticklabels=sorted(y.unique()))
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix (Class-Balanced)')
# plt.tight_layout()
# plt.show()
#
# # 4.8 (Answered in text document)
# # 4.9 - Variable importance
# for feature, coef in zip(X_train.columns, model.coef_[0]):
#     print(f"{feature}: {coef:.4f}")




# Problem 5
# 5.1 Data partitioning
df = pd.read_csv('powdermetallurgy.csv')

print("total rows:", len(df))
print("unique values:", df['Shrinkage'].unique())
print("data types:", df.dtypes)  # by default, get_dummies encodes object-type or categorical columns into binary ones

df = pd.get_dummies(df) # must dummy encode variables for use with linear regression
X = df.drop('Shrinkage', axis=1)  # axis 1 specifies we're dropping a column, not a row
y = df['Shrinkage']

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=1/3, random_state=42)

# 5.2 Fit multiple linear regression, predict, compute MAE and RMSE
model = LinearRegression()  # same as used for simple linear regression, adaptable
model.fit(X_train, y_train)

y_pred = model.predict(X_holdout)

mae = mean_absolute_error(y_holdout, y_pred)
print(f'MAE): {mae:.4f}')

rmse = root_mean_squared_error(y_holdout, y_pred)
print(f"RMSE: {rmse:.4f}")

# Problem 6
intercept = -3.485
coef_A1 = 0.045
coef_A2 = 0.003

def logistic_regression(A1, A2):
    # Calculate the logit (linear combination of features and coefficients)
    logit = intercept + (coef_A1 * A1) + (coef_A2 * A2)

    # Apply the sigmoid function to get the probability
    p = 1 / (1 + math.exp(-logit))

    return p


def classify(p):
    if p >= 0.5:
        return "yes"
    else:
        return "no"

# Object O1: A1 = 47, A2 = 213
p_O1 = logistic_regression(47, 213)
classification_O1 = classify(p_O1)
print(f"Object O1 classified as: {classification_O1} (Probability: {p_O1:.4f})")

# Object O2: A1 = 65, A2 = 276
p_O2 = logistic_regression(65, 276)
classification_O2 = classify(p_O2)
print(f"Object O2 classified as: {classification_O2} (Probability: {p_O2:.4f})")