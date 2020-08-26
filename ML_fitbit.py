#import libraries
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
from sklearn.tree.export import export_text
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#parser function
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
#read relevant csv
new_df = pd.read_csv('combined_csv_files_{}_final_df.csv'.format(args.name), header = 0)

#convert dateTime column type to datetime
new_df['dateTime'] = pd.to_datetime(new_df['dateTime'])


#Machine Learning
#use LabelEncoder to convert day names to categorical values
number = LabelEncoder()
new_df['day_num'] = number.fit_transform(new_df['day'].astype("str"))

print(args.train)

#add feature that will easily allow either for the building of the model to predict calories or for the building of the model to predict total steps done
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

#Decision Tree
#split the data in train dataframe and test dataframe with the ratio 0.7:0.3
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
#print shape of train and test dataframes
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

print(y_train)
print(x_train.describe())

#train decision tree on train dataframe
clf_final = tree.DecisionTreeClassifier(random_state=42)
clf_final.fit(x_train, y_train)
#predict target test values using test dataframe
y_pred = clf_final.predict(x_test)

#set up matplot figure
fig, ax = plt.subplots(figsize=(8, 8))
#plot the decision tree
tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
plt.savefig('{}/decision_tree_{}.png'.format(path, args.train), dpi=400) #save the figure
#alternative way to plot the decision tree by manually entering the size and colour of the arrows
# out = tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
# for o in out:
#     arrow = o.arrow_patch
#     if arrow is not None:
#         arrow.set_edgecolor('black')
#         arrow.set_linewidth(3)

#plot and output the decision tree as text
tree_rules = export_text(clf_final, feature_names=list(X.columns))
print(tree_rules)

#extract feature importances
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': clf_final.feature_importances_}).\
                    sort_values('importance', ascending = False)

print(fi.head())

#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot bar chart showing feature importances
sns.barplot(x=fi.feature, y=fi.importance)
#add labels to the graph
plt.xlabel('Features') #rename the x-axis title
plt.ylabel('Feature Importance Score') #rename the y-axis title
plt.title("Visualizing Important Features") #rename the title of the figure
plt.xticks(rotation=75) #rotate the x-axis labels
plt.legend() #add legend
fig.savefig('{}/vis_features_{}.png'.format(path, args.train), dpi=400) #save the figure

#compute accuracy of the decision tree using train dataframe
train_accuracy = clf_final.score(x_train, y_train) 
#compute accuracy of the decision tree using test dataframe
val_accuracy = clf_final.score(x_test, y_test) 
#print accuracies
print('train accuracy is {}, val_accuracy is {}'.format(train_accuracy,val_accuracy))

#alternative way to compute accuracy of the decision tree using test daraframe
accuracy = accuracy_score(y_test, y_pred)
#print accuracy
print('y_test {}'.format(y_test))

#find values of false positive rates and true positive rates that will be used to plot the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
#compute the area under the ROC curve
auc = metrics.roc_auc_score(y_test, y_pred)
#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot the ROC curve
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#add legend
plt.legend(loc=4)
plt.savefig('{}/ROC_curve_{}.png'.format(path, args.train), dpi=400) #save the figure

#Find max depth with best accuracy
#split the data in train dataframe, validation dataframe and test dataframe with the ratio 0.7:0.15:0.15
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

print(len(x_train))
print(len(x_val))
print(len(x_test))

#create dictionaries
train_accuracy_dict = dict()
val_accuracy_dict = dict()
val_recall_dict = dict()
val_precision_dict= dict()

#use for loop to train different decision trees on different maximum depths using train dataframe
#compute the accuracy of each decision tree (with different maximum depths) using validation dataframe
#store the results in the dictionaries created above
for i in range (1,53):
    clfi = tree.DecisionTreeClassifier(random_state=42, max_depth = i)
    clfi = clfi.fit(x_train, y_train) 
    train_accuracy_dict[i] = clfi.score(x_train, y_train)
    val_predict = clfi.predict(x_val)
    val_accuracy_dict[i] = clfi.score(x_val, y_val)
    val_recall_dict[i] = recall_score(val_predict, y_val)
    val_precision_dict[i] = precision_score(val_predict, y_val)

#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot maximum depth against the accuracy of each decision tree using train dataframe
plt.plot(list(train_accuracy_dict.keys()), list(train_accuracy_dict.values()), color='red')
#plot maximum depth against the accuracy of each decision tree using validation dataframe
plt.plot(list(val_accuracy_dict.keys()), list(val_accuracy_dict.values()), color='blue')
#add legend
plt.gca().legend(('training','validation')) 
plt.ylabel('Accuracy') #rename the y-axis title
plt.xlabel('Max_depth') #rename the x-axis title
fig.savefig('{}/max_depth_accuracy_{}.png'.format(path, args.train), dpi=400) #save the figure

#find the decision tree and maximum depth of specific decision tree with maximum accuracy using validation dataframe
depth_optimal=max(val_accuracy_dict, key=val_accuracy_dict.get)
#print the optimal maximum depth along with the accuracy of the specific decision tree
print(val_accuracy_dict[depth_optimal],depth_optimal)

#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot maximum depth against recall values of each decision tree
plt.plot(list(val_recall_dict.keys()), list(val_recall_dict.values()), color='red', label = 'Recall')
#plot maximum depth against precision values of each decision tree
plt.plot(list(val_precision_dict.keys()), list(val_precision_dict.values()), color='blue', label = 'Precision')
#add legend
plt.legend()
fig.savefig('{}/recall_precision_{}.png'.format(path, args.train), dpi=400) #save the figure

#merge train and validation dataframe
x_trainfull = x_train.append(x_val)
y_trainfull = y_train.append(y_val)

#train decision tree on train and validation dataframe using optimal maximum depth (found above)
clf_full = tree.DecisionTreeClassifier(random_state=42, max_depth=depth_optimal)
clf_full = clf_full.fit(x_trainfull, y_trainfull)
#compute accuracy of the decision tree using train and validation dataframe
trainfull_accuracy = clf_full.score(x_trainfull, y_trainfull) 
#compute accuracy of the decision tree using test dataframe
testfull_accuracy = clf_full.score(x_test, y_test) 
[trainfull_accuracy, testfull_accuracy]
#print accuracies
print('trainfull accuracy is {}, valfull_accuracy is {}'.format(trainfull_accuracy,testfull_accuracy))

#print recall scores of decision tree using test dataframe
print(recall_score(clf_full.predict(x_test), y_test))
#plot confusion matrix
cm = plot_confusion_matrix(clf_full, x_test, y_test, values_format='2g') 
plt.savefig('{}/confusion_{}.png'.format(path, args.train), dpi=400) #save the figure

#plot the decision tree 
tree.plot_tree(clf_full, feature_names=X.columns, class_names=['No', 'Yes'], filled=True) 

#plot and output the decision tree as text
r=export_text(clf_final, feature_names=list(X.columns))
print(r)

#predict target test values using test dataframe
y_pred = clf_full.predict(x_test)
#find values of false positive rates and true positive rates that will be used to plot the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
#compute the area under the ROC curve
auc = metrics.roc_auc_score(y_test, y_pred)
#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot the ROC curve
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#add legend
plt.legend(loc=4)
fig.savefig('{}/ROC_curve2_{}.png'.format(path, args.train), dpi=400) #save the figure

#Random Forest
#split the data in train dataframe and test dataframe with the ratio 0.7:0.3
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
#print shape of train and test dataframes
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

#instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
#train the model using train dataframe
rf.fit(x_train, y_train)
#compute and output accuracy of the model using test dataframe
print('rf_score is {}'.format(rf.score(x_test, y_test)))

#predict target test values using test dataframe
rf_predictions = rf.predict(x_test)
#probabilities for each class
rf_probs = rf.predict_proba(x_test)[:,1]

#find values of false positive rates and true positive rates that will be used to plot the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,  rf_probs)
#compute the area under the ROC curve
auc = metrics.roc_auc_score(y_test, rf_probs)
#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot the ROC curve
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#add legend
plt.legend(loc=4)
fig.savefig('{}/ROC_curve3_{}.png'.format(path, args.train), dpi=400) #save the figure

#extract feature importances
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': rf.feature_importances_}).\
                    sort_values('importance', ascending = False)

print(fi.head())

#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot bar chart showing feature importances
sns.barplot(x=fi.feature, y=fi.importance)
#add labels to the graph
plt.xlabel('Features') #rename the x-axis title
plt.ylabel('Feature Importance Score') #rename the y-axis title
plt.title("Visualizing Important Features") #rename the title of the figure
plt.xticks(rotation=75) #rotate the x-axis labels
plt.legend() #add legend
fig.savefig('{}/vis_feat_2_{}.png'.format(path, args.train), dpi=400) #save the figure

#pull out one tree from the forest (tree number 5)
tree = rf.estimators_[5]
#export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X.columns, rounded = True, precision = 1)
#use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
#write the graph to a png file
graph.write_png('tree.png')

#Logistic Regression
#print shape of dataframes
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

#split the data in train dataframe and test dataframe with the ratio 0.7:0.3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
#print shape of train and test dataframes
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

#train logistic regresion
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
#print the coefficients along with statistical details regarding the coefficients (e.g. how effective they are in predicting the target value)
print(result.summary2())

#train logistic regression on train dataframe
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

#predict target test values using test dataframe
y_pred = logreg.predict(x_test)
#compute and output accuracy of the model using test dataframe
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

#find values of false positive rates and true positive rates that will be used to plot the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
#compute the area under the ROC curve
auc = metrics.roc_auc_score(y_test, y_pred)
#set up matplot figure
fig = plt.figure(figsize=(8, 8))
#plot the ROC curve
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#add legend
plt.legend(loc=4)
fig.savefig('{}/ROC_curve4_{}.png'.format(path, args.train), dpi=400) #save the figure

#Linear Regression
#add feature that will easily allow either for the building of the model to predict calories or for the building of the model to predict total steps done
if args.train == "calories":
    X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
            'very_active_minutes', 'overall_score', 'resting_heart_rate']]
    Y = new_df['Total_Calories']
elif args.train == "steps":
    X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
            'very_active_minutes', 'overall_score', 'resting_heart_rate']]
    Y = new_df['Total_Steps']  

#fill empty values with the mean value
X.fillna(X.mean(), inplace=True)

#print shape of dataframes
print('X shape: {}'.format(X.shape))
print('Y shape: {}'.format(Y.shape))

#split the data in train dataframe and test dataframe with the ratio 0.7:0.3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
#print shape of train and test dataframes
print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))

#create linear regression object
regr = linear_model.LinearRegression()
#train the model using training dataframe
regr.fit(x_train, y_train)

#predict target test values using test dataframe
lr_y_pred = regr.predict(x_test)

#find coefficients of linear regression
coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(regr.coef_))], axis = 1)
#print coefficients
print(coefficients)
#compute and output r2 error using test dataframe
print("r2_error: %.2f"% r2_score(y_test, lr_y_pred))
#compute and output mean squared error using test dataframe
print("Mean squared error: %.2f"% mean_squared_error(y_test, lr_y_pred))
#compute and output mean absolute error using test dataframe
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lr_y_pred))  
#compute and output root mean squared error using test dataframe
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lr_y_pred)))

#create dataframe with actual and predicted target values
df = pd.DataFrame({'Actual': y_test, 'Predicted': lr_y_pred})
#print the top 25 rows of the dataframe
print(df.head(25))

#print the coefficients along with statistical details regarding the coefficients (e.g. how effective they are in predicting the target value)
print(stats.summary(regr, x_train, y_train, x_train.columns))
#print('THE END')
