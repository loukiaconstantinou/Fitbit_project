import pandas as pd
import os
from functools import reduce
import ast
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from regressors import stats

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Input data path (dataframe_csv)", default= os.getcwd())
    parser.add_argument("--train", type=str, help="choose between steps and calories", choices= ["steps", "calories"], default= "calories")
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args

args = parse_args()

if not os.path.isdir("plots_{}".format(args.name)):
    os.makedirs("plots_{}".format(args.name))

path = 'plots_{}'.format(args.name)
new_df = pd.read_csv('combined_csv_files_{}_final_df.csv'.format(args.name), header = 0)

new_df['dateTime'] = pd.to_datetime(new_df['dateTime'])


# ## Machine Learning

number = LabelEncoder()
new_df['day_num'] = number.fit_transform(new_df['day'].astype("str"))

print(args.train)

if args.train == "calories":
    X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
           'very_active_minutes', 'overall_score', 'resting_heart_rate']]
    threshold = 2000
    Y = new_df['Total_Calories'] > threshold
elif args.train == "steps":
    X = new_df[['day_num', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
            'very_active_minutes', 'overall_score', 'resting_heart_rate', 'Total_Calories']]
    Y = new_df['success_steps'] == 'Y'

print(new_df.dtypes)
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

print(y_train)
print(x_train.describe())

clf_final = tree.DecisionTreeClassifier(random_state=42)
clf_final.fit(x_train, y_train)
y_pred = clf_final.predict(x_test)

fig, ax = plt.subplots(figsize=(8, 8))
tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
plt.savefig('{}/decision_tree_{}.png'.format(path, args.train), dpi=400)
# out = tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
# for o in out:
#     arrow = o.arrow_patch
#     if arrow is not None:
#         arrow.set_edgecolor('black')
#         arrow.set_linewidth(3)


from sklearn.tree.export import export_text
tree_rules = export_text(clf_final, feature_names=list(X.columns))
print(tree_rules)


# Extract feature importances
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': clf_final.feature_importances_}).\
                    sort_values('importance', ascending = False)

print(fi.head())

# Creating a bar plot
fig = plt.figure(figsize=(8, 8))
sns.barplot(x=fi.feature, y=fi.importance)
# Add labels to your graph
plt.xlabel('Features')
plt.ylabel('Feature Importance Score')
plt.title("Visualizing Important Features")
plt.xticks(rotation=75)
plt.legend()
fig.savefig('{}/vis_features_{}.png'.format(path, args.train), dpi=400)

train_accuracy = clf_final.score(x_train, y_train) 
val_accuracy = clf_final.score(x_test, y_test) 
print('train accuracy is {}, val_accuracy is {}'.format(train_accuracy,val_accuracy))


accuracy = accuracy_score(y_test, y_pred)
print('y_test {}'.format(y_test))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
fig = plt.figure(figsize=(8, 8))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig('{}/ROC_curve_{}.png'.format(path, args.train), dpi=400)

#Find max depth with best accuracy

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


print(len(x_train))
print(len(x_val))
print(len(x_test))

train_accuracy_dict = dict()
val_accuracy_dict = dict()
val_recall_dict = dict()
val_precision_dict= dict()

for i in range (1,53):
    clfi = tree.DecisionTreeClassifier(random_state=42, max_depth = i)
    clfi = clfi.fit(x_train, y_train) 
    train_accuracy_dict[i] = clfi.score(x_train, y_train)
    val_predict = clfi.predict(x_val)
    val_accuracy_dict[i] = clfi.score(x_val, y_val)
    val_recall_dict[i] = recall_score(val_predict, y_val)
    val_precision_dict[i] = precision_score(val_predict, y_val)


fig = plt.figure(figsize=(8, 8))
plt.plot(list(train_accuracy_dict.keys()), list(train_accuracy_dict.values()), color='red')
plt.plot(list(val_accuracy_dict.keys()), list(val_accuracy_dict.values()), color='blue')
plt.gca().legend(('training','validation'))
plt.ylabel('Accuracy')
plt.xlabel('Max_depth')
fig.savefig('{}/max_depth_accuracy_{}.png'.format(path, args.train), dpi=400)

depth_optimal=max(val_accuracy_dict, key=val_accuracy_dict.get)
print(val_accuracy_dict[depth_optimal],depth_optimal)

fig = plt.figure(figsize=(8, 8))
plt.plot(list(val_recall_dict.keys()), list(val_recall_dict.values()), color='red', label = 'Recall')
plt.plot(list(val_precision_dict.keys()), list(val_precision_dict.values()), color='blue', label = 'Precision')
plt.legend()
fig.savefig('{}/recall_precision_{}.png'.format(path, args.train), dpi=400)



x_trainfull = x_train.append(x_val)
y_trainfull = y_train.append(y_val)


clf_full = tree.DecisionTreeClassifier(random_state=42, max_depth=depth_optimal)
clf_full = clf_full.fit(x_trainfull, y_trainfull)
trainfull_accuracy = clf_full.score(x_trainfull, y_trainfull) 
testfull_accuracy = clf_full.score(x_test, y_test) 
[trainfull_accuracy, testfull_accuracy]
print('trainfull accuracy is {}, valfull_accuracy is {}'.format(trainfull_accuracy,testfull_accuracy))



from sklearn.metrics import plot_confusion_matrix
#recall score shows the ability of the classifier to find all positive values: tp / (tp+fn)
print(recall_score(clf_full.predict(x_test), y_test))
cm = plot_confusion_matrix(clf_full, x_test, y_test, values_format='2g')  # <----------------
# plt.show(cm)
plt.savefig('{}/confusion_{}.png'.format(path, args.train), dpi=400)

tree.plot_tree(clf_full, feature_names=X.columns, class_names=['No', 'Yes'], filled=True) # <----------------


from sklearn.tree.export import export_text
r=export_text(clf_final, feature_names=list(X.columns))
print(r)

y_pred = clf_full.predict(x_test)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
fig = plt.figure(figsize=(8, 8))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
fig.savefig('{}/ROC_curve2_{}.png'.format(path, args.train), dpi=400)

#Random Forest


x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)
print('rf_score is {}'.format(rf.score(x_test, y_test)))

# Actual class predictions
rf_predictions = rf.predict(x_test)
# Probabilities for each class
rf_probs = rf.predict_proba(x_test)[:,1]

y_pred = clf_full.predict(x_test)
fpr, tpr, _ = metrics.roc_curve(y_test,  rf_probs)
auc = metrics.roc_auc_score(y_test, rf_probs)
fig = plt.figure(figsize=(8, 8))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
fig.savefig('{}/ROC_curve3_{}.png'.format(path, args.train), dpi=400)

fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)

print(fi.head())

fig = plt.figure(figsize=(8, 8))
# Creating a bar plot
sns.barplot(x=fi.feature, y=fi.importance)
# Add labels to your graph
plt.xlabel('Features')
plt.ylabel('Feature Importance Score')
plt.title("Visualizing Important Features")
plt.xticks(rotation=75)
plt.legend()
fig.savefig('{}/vis_feat_2_{}.png'.format(path, args.train), dpi=400)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')



###Logistic Regression


print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

import statsmodels.api as sm

logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
fig = plt.figure(figsize=(8, 8))
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
fig.savefig('{}/ROC_curve4_{}.png'.format(path, args.train), dpi=400)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


##LINEAR REGRESSION

if args.train == "calories":
    X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
            'very_active_minutes', 'overall_score', 'resting_heart_rate']]
    Y = new_df['Total_Calories']
elif args.train == "steps":
    X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
            'very_active_minutes', 'overall_score', 'resting_heart_rate']]
    Y = new_df['Total_Steps']  

X.fillna(X.mean(), inplace=True)

print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
lr_y_pred = regr.predict(x_test)

coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(regr.coef_))], axis = 1)
print(coefficients)
print("r2_error: %.2f"% r2_score(y_test, lr_y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, lr_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lr_y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lr_y_pred)))

df = pd.DataFrame({'Actual': y_test, 'Predicted': lr_y_pred})
print(df.head(25))

print(stats.summary(regr, x_train, y_train, x_train.columns))
print('THE END')

