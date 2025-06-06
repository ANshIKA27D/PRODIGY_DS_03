import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#======================================================================================

data = pd.read_csv(r"C:\Users\ANJALI DUBEY\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional.csv", sep=';')
print(data.head())
print(data.info())

data.replace('unknown', np.nan, inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)
#--------------------------------------
data.hist(figsize=(10, 10))
plt.tight_layout()
plt.show()

#-----------------------------------------
numeric_col = data.select_dtypes(include=[np.number])
corr = numeric_col.corr()
sns.heatmap(corr, annot=True, cmap='cividis', linewidths=0.2)
plt.show()
#=============================================
#Encode target variable ('y') to binary values
y = data['y'].map({'no': 0, 'yes': 1})
X = data.drop('y', axis=1)
X = pd.get_dummies(X, drop_first=True)

#-----------------------------------
#Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
#=====================================================================
importances = clf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(10))
feat_imp.plot(kind='bar', figsize=(10, 5))
plt.title("Feature Importances")
plt.show()

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average CV score:", np.mean(scores))
#==================================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot()
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(clf, max_depth=3, filled=True, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], rounded=True, fontsize=10)
plt.title("Decision Tree (First 3 Levels)")
plt.show()
