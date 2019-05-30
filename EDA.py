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

def PCA_plot(X_plot, y_plot):
	# PCA for data visualization
	estimator = PCA(n_components=2)
	X_pca = estimator.fit_transform(X_plot)
	colors = []
	for i in range(len(y_plot)):
		if y_plot.iloc[i]==1:
			colors.append('orange')
		else:
			colors.append('blue')
		plt.scatter(X_pca[i,0], X_pca[i,1], c=colors[i])
	plt.show()


def plot_labels(y_plot):
	plt.figure()
	plt.hist(y_plot, color='darkorange')
	plt.xlabel('Values')
	plt.ylabel('Frequency')
	plt.title('Histogram of balanced target labels')
	plt.show()

def plot_ROC(y_true, y_score):
	#Plot ROC curve
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
	#AUC calculation
	roc_auc = metrics.auc(fpr, tpr)
	print("AUC:", roc_auc)

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


# Opens csv file
secom = pd.read_csv('uci-secom.csv')
# Overview of data
print(secom)  
# Caracteristics as mean, min and max values    
secom.describe() 
# Data types of each column
secom.dtypes 
# Print shape
print("shape of the entire dataset:")
print(secom.shape)
secom.info()

# Random undersampling
secom_pass = secom[secom['Pass/Fail'] == 1]
secom_fail = secom[secom['Pass/Fail'] == -1]

count_pass = secom_pass.shape[0]
count_fail = secom_fail.shape[0]

secom_fail = secom.sample(int(round(10*count_pass)), random_state=123)

secom_resampled = pd.concat([secom_pass, secom_fail], axis=0)

# Split into dataset and label
X, y = secom_resampled.iloc[:,1:-1], secom_resampled.iloc[:,-1]

# Replace nul values for Nan
for column in X:
	X[column].replace(0.0, np.nan, inplace= True)

# Replace missing values (NaN)
for column in X:
#	X[column].replace(0.0, np.nan, inplace= True)
	mean = X[column].mean()
	if np.isnan(mean):
		X = X.drop([column],axis=1)
	else:
		X[column].fillna(mean, inplace=True)

# print("X shape after eliminating NaN values:")
# print(X.shape)


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 				#Splits to 20% of the data for testing, separating in classes according to target labels
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)


# Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), 				#Pipeline that first fits the data by Calling the transformer API 
                         RandomForestClassifier(class_weight='balanced', random_state=123, n_estimators=50))		#and then models it throught RandomForestRegressor

################################## Uncomment when tunning hyperparameters
# Declare hyperparameters to tune
# hyperparameters = { 															#Dictionary structure
# 				  'randomforestclassifier__max_features' : ['auto', 'log2'],		
#                   'randomforestclassifier__max_depth': [None, 5, 3, 1],
#                   'randomforestclassifier__n_estimators' : range(10,100,10),		
#                   'randomforestclassifier__criterion': ['gini', 'entropy']
#                   'randomforestclassifier__min_samples_leaf': range(1,10),
#     			  'randomforestclassifier__min_samples_split': range(2,20,2)
#                   }

# Tune model using cross-validation pipeline
# clf = GridSearchCV(pipeline, hyperparameters, cv=3)					#(model, hyperprameters, number of folds to create)
# clf.fit(X_train, y_train)
# 
# print("Best parameters:")
# print(clf.best_params_)
# best_clf = clf.best_estimator_

# pipeline.fit(X_train, y_train)
# y_pred_base = pipeline.predict(X_test)

# y_pred_base_pos = y_pred_base[y_pred_base == 1]
# y_test_pos = y_test[y_test == 1]
# print("Predicted base posive and real positive")
# print(y_pred_base_pos.sum(), y_test_pos.sum())

# y_pred_base_neg = y_pred_base[y_pred_base == -1]
# y_test_neg = y_test[y_test == -1]
# print("Predicted base negative and real negative")
# print(-y_pred_base_neg.sum(), -y_test_neg.sum())

################################## Uncomment when NOT tunning hyperparameters
clf = pipeline
clf.fit(X_train, y_train)

# Creates a dataset with features importances ordered by bigger importance
feature_importances = pd.DataFrame(clf.steps[1][1].feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances.head(10))

# # Evaluate model pipeline on test data
y_pred = clf.predict(X_test)

#Print metrics
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred, labels=[1,-1]))
print("Classification Report:", metrics.classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F1 total score:", metrics.f1_score(y_test, y_pred, average='macro'))

#Plot ROC curve
plot_ROC(y_test, y_pred)


#########################################################################################################################################################
# 3. Selecting features by importances
model = feature_selection.SelectFromModel(clf.steps[1][1], prefit=True, threshold=0.0004)
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)

print("X new shape after feature selection:", X_train_new.shape, X_test_new.shape)

clf.fit(X_train_new, y_train)

# # Evaluate model pipeline on test data
y_pred = clf.predict(X_test_new)

#Print metrics
print("Confusion matrix after feature selection:")
print(metrics.confusion_matrix(y_test, y_pred, labels=[1,-1]))
print("Classification Report after feature selection:", metrics.classification_report(y_test, y_pred))
print("Accuracy feature selection:",metrics.accuracy_score(y_test, y_pred))

#Print F1 score
print("F1 total score after feature selection:", metrics.f1_score(y_test, y_pred, average='macro'))

#Plot ROC curve
plot_ROC(y_test, y_pred)
