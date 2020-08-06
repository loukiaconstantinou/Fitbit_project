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
import matplotlib.pyplot as plt

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

#X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
 #           'very_active_minutes', 'overall_score', 'resting_heart_rate']]
#X = new_df[['day_num', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
 #           'very_active_minutes', 'overall_score', 'resting_heart_rate']]
#X.fillna(X.mean(), inplace=True)

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

clf_final = tree.DecisionTreeClassifier(random_state=42)
clf_final.fit(x_train, y_train)
y_pred = clf_final.predict(x_test)

fig, ax = plt.subplots(figsize=(8, 8))
tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
plt.show()
plt.savefig('{}/decision_tree_{}.png'.format(path, args.train), dpi=400)
# out = tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
# for o in out:
#     arrow = o.arrow_patch
#     if arrow is not None:
#         arrow.set_edgecolor('black')
#         arrow.set_linewidth(3)


from sklearn.tree.export import export_text
tree_rules = export_text(clf_final, feature_names=list(x_train))
print(tree_rules)

train_accuracy = clf_final.score(x_train, y_train) 
val_accuracy = clf_final.score(x_test, y_test) 
print('train accuracy is {}, val_accuracy is {}'.format(train_accuracy,val_accuracy))


accuracy = accuracy_score(y_test, y_pred)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
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
plt.show()
fig.savefig('{}/max_depth_accuracy_{}.png'.format(path, args.train), dpi=400)

depth_optimal=max(val_accuracy_dict, key=val_accuracy_dict.get)
print(val_accuracy_dict[depth_optimal],depth_optimal)


# # In[104]:


# plt.plot(list(val_recall_dict.keys()), list(val_recall_dict.values()), color='red', label = 'Recall')
# plt.plot(list(val_precision_dict.keys()), list(val_precision_dict.values()), color='blue', label = 'Precision')
# plt.legend()


# # In[105]:


# x_trainfull = x_train.append(x_val)
# y_trainfull = y_train.append(y_val)


# # In[106]:


# clf_full = tree.DecisionTreeClassifier(random_state=42, max_depth=depth_optimal)
# clf_full = clf_full.fit(x_trainfull, y_trainfull)
# trainfull_accuracy = clf_full.score(x_trainfull, y_trainfull) 
# testfull_accuracy = clf_full.score(x_test, y_test) 
# [trainfull_accuracy, testfull_accuracy]


# # In[107]:


# from sklearn.metrics import plot_confusion_matrix

# print(recall_score(clf_full.predict(x_test), y_test))
# plot_confusion_matrix(clf_full, x_test, y_test, values_format='2g')


# # In[108]:


# clf_final = tree.DecisionTreeClassifier(random_state=42, max_depth=depth_optimal)
# clf_final.fit(X, Y)
# tree.plot_tree(clf_final, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)


# # In[109]:


# from sklearn.tree.export import export_text
# r=export_text(clf_final, feature_names=list(X.columns))
# print(r)


# # In[110]:


# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(x_train, y_train)

# # Make predictions using the testing set
# lr_y_pred = regr.predict(x_test)
                        
# print("Mean squared error: %.2f"% mean_squared_error(y_test, lr_y_pred))
# coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(regr.coef_))], axis = 1)
# print(coefficients)
# print("r2_error: %.2f"% r2_score(y_test, lr_y_pred))


# # In[111]:


# y_train


# # In[152]:


# X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
#             'very_active_minutes', 'overall_score', 'resting_heart_rate']]
# X.fillna(X.mean(), inplace=True)

# #threshold = 2000

# #Y = new_df['Total_Calories'] > threshold
# Y = new_df['success_steps']

# print('X shape: {}'.format(X.shape))
# print('Y shape: {}'.format(Y.shape))

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
# print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))


# # In[142]:


# import statsmodels.api as sm
# logit_model=sm.Logit(Y,X)
# result=logit_model.fit()
# print(result.summary2())


# # In[153]:


# from sklearn.linear_model import LogisticRegression

# logreg = LogisticRegression()
# logreg.fit(x_train, y_train)

# y_pred = logreg.predict(x_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# # In[155]:


# y_test = y_test.replace('Y',1)
# y_test = y_test.replace('N',0)


# # In[156]:


# y_test= 1 <= y_test
# print(y_test)


# # In[157]:


# from sklearn import metrics

# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
# auc = metrics.roc_auc_score(y_test, y_pred)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()


# # In[140]:


# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
# fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()


# # In[144]:


# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


# # In[ ]:




