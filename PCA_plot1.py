import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import feature_selection
from imblearn.under_sampling import TomekLinks

# Opens csv file
secom = pd.read_csv('uci-secom.csv')

X, y = secom.iloc[:,1:-1], secom.iloc[:,-1]

print("Shape", X.shape)
# Replace nul values for Nan
for column in X:
	X[column].replace(0.0, np.nan, inplace= True)
#print(X.isnull().sum().sum())						#Returned 241956

# Replace missing values (NaN)
for column in X:
#	X[column].replace(0.0, np.nan, inplace= True)
	mean = X[column].mean()
	if np.isnan(mean):
		X = X.drop([column],axis=1)
	else:
		X[column].fillna(mean, inplace=True)
#print(X.isnull().sum().sum())
print("New shape", X.shape)
# # Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 				#Splits to 20% of the data for testing, separating in classes according to target labels
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)


# Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), 				#Pipeline that first fits the data by Calling the transformer API 
                         RandomForestClassifier(class_weight='balanced', random_state=123, n_estimators=50))		#and then models it throught RandomForestRegressor

#Uncoment when not tunning parameters
clf = pipeline
clf.fit(X_train, y_train)
# Creates a dataset with features importances ordered by bigger importance
feature_importances = pd.DataFrame(clf.steps[1][1].feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

# # Evaluate model pipeline on test data
y_pred = clf.predict(X_test)

y_pred_pos = y_pred[y_pred == 1]
y_test_pos = y_test[y_test == 1]
print("Predicted posive and real positive")
print(y_pred_pos.sum(), y_test_pos.sum())

y_pred_neg = y_pred[y_pred == -1]
y_test_neg = y_test[y_test == -1]
print("Predicted negative and real negative")
print(-y_pred_neg.sum(), -y_test_neg.sum())

print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred, labels=[1,-1]))
print("Classification Report")
print(metrics.classification_report(y_test, y_pred))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision, Recall, F1 score:", metrics.precision_recall_fscore_support(y_test, y_pred, labels=[1,-1]))
#Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, linestyle='-',
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.025, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

secom_pass = secom[secom['Pass/Fail'] == 1]
secom_fail = secom[secom['Pass/Fail'] == -1]


plt.figure()
plt.bar(secom_pass, color='darkorange', secom_fail, color='navy')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Bar plot of target labels')
plt.show()
