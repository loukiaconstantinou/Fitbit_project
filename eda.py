
# In[59]:


# new_df.columns


# # In[60]:


# new_df.head()

# ################
# # ## EDA

# # In[61]:


# corrdf_calories = new_df[['Total_Calories','Total_Steps', 'is_weekend', 'sedentary_minutes', 'very_active_minutes', 
#                             'moderately_active_minutes', 'lightly_active_minutes', 'resting_heart_rate', 'overall_score']]

# import seaborn as sns
# import matplotlib.pyplot as plt
# def CorrMtx(df, dropDuplicates = True):

#     # Your dataset is already a correlation matrix.
#     # If you have a dateset where you need to include the calculation
#     # of a correlation matrix, just uncomment the line below:
#     df = df.corr()

#     # Exclude duplicate correlations by masking uper right values
#     if dropDuplicates:    
#         mask = np.zeros_like(df, dtype=np.bool)
#         mask[np.triu_indices_from(mask)] = True

#     # Set background color / chart style
#     sns.set_style(style = 'white')

#     # Set up  matplotlib figure
#     f, ax = plt.subplots(figsize=(11, 9))

#     # Add diverging colormap from red to blue
#     cmap = sns.diverging_palette(250, 10, as_cmap=True)
#     # Draw correlation plot with or without duplicates
#     if dropDuplicates:
#         sns.heatmap(df, mask=mask, cmap=cmap, 
#                 square=True,
#                 linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
#     else:
#         sns.heatmap(df, cmap=cmap, 
#                 square=True,
#                 linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


# CorrMtx(corrdf_calories, dropDuplicates = False)


# # In[62]:


# sns.pairplot(corrdf_calories.dropna(), kind="scatter", markers="+", plot_kws=dict(s=50, edgecolor="b", linewidth=1))
# plt.show()


# # In[63]:


# fig = plt.figure(figsize = (20,6))

# ax = plt.subplot(141)  
# new_df.groupby('day').Total_Steps.mean().plot.bar()
# plt.title('Day of Week vs. Steps', fontsize=15)
# plt.xlabel('Day of Week', fontsize=14)
# plt.ylabel('Steps', fontsize=14)
# ax.axhline(8000, color="orangered", linestyle='--')
# ax.axhline(10000, color="orange", linestyle='--')

# ax2 = fig.add_subplot(142)
# new_df.groupby('day').Total_Calories.mean().plot.bar()
# plt.title('Day of Week vs. Calories Burned', fontsize=15)
# plt.xlabel('Day of Week', fontsize=14)
# plt.ylabel('Calories Burned', fontsize=14)

# ax3 = fig.add_subplot(143)
# new_df.groupby('day').very_active_minutes.mean().plot.bar()
# plt.title('Day of Week vs. Minutes Very Active', fontsize=15)
# plt.xlabel('Day of Week', fontsize=14)
# plt.ylabel('Minutes Very Active', fontsize=14)

# ax4 = fig.add_subplot(144)
# new_df.groupby('day').sedentary_minutes.mean().plot.bar()
# plt.title('Day of Week vs. Minutes Sedentary', fontsize=15)
# plt.xlabel('Day of Week', fontsize=14)
# plt.ylabel('Minutes Sedentary', fontsize=14)


# # In[65]:


# new_df.groupby('day').resting_heart_rate.mean().plot.line()
# plt.title('Day of Week vs. Resting Heart Rate', fontsize=15)
# plt.xlabel('Day of Week', fontsize=14)
# plt.ylabel('Resting Heart Rate', fontsize=14)


# # In[66]:


# new_df['minutes_asleep'] = new_df['deep'] + new_df['wake'] + new_df['light'] + new_df['rem']


# # In[67]:


# new_df = new_df.replace("", 0)


# # In[68]:


# new_df = new_df.reset_index()


# # In[69]:


# for i in range(len(new_df)):
#     if new_df['minutes_asleep'][i] == 0:
#         new_df['minutes_asleep'][i] = new_df['asleep'][i]


# # In[70]:


# new_df.head()


# # In[71]:


# row = ((new_df.loc[new_df['deep'] == 0]))


# # In[72]:


# sleep_perc_df = new_df[~new_df.index.isin(row.index)]


# # In[73]:


# sleep_perc_df['deep_perc'] = sleep_perc_df['deep']/sleep_perc_df['minutes_asleep']
# sleep_perc_df['wake_perc'] = sleep_perc_df['wake']/sleep_perc_df['minutes_asleep']
# sleep_perc_df['light_perc'] = sleep_perc_df['light']/sleep_perc_df['minutes_asleep']
# sleep_perc_df['rem_perc'] = sleep_perc_df['rem']/sleep_perc_df['minutes_asleep']


# # In[74]:


# avg_perc_sleep = sleep_perc_df[['deep_perc', 'wake_perc', 'light_perc', 'rem_perc']].mean()

# fig = plt.figure(figsize = (6,6))
# labels=['Deep sleep', 'Awake', 'Light sleep', 'REM sleep']
# plt.pie(avg_perc_sleep, colors = ['darkturquoise', 'salmon', 'lightskyblue', 'yellowgreen'], autopct='%1.1f%%', labels=labels, textprops=dict(color="w"))

# # #carve the donut
# # my_circle=plt.Circle( (0,0), 0.7, color='white')
# # p=plt.gcf()
# # p.gca().add_artist(my_circle)

# plt.title('Average of types of sleep', fontsize=14)
# plt.legend()
# plt.show()


# # In[75]:


# avg_perc_sleep


# # In[76]:


# sleep_perc_df.groupby('day').mean()[["deep", "wake", "light", "rem"]].plot(kind='bar')


# # In[77]:


# f = plt.figure(figsize=(15,5))

# ax = f.add_subplot(121)
# new_df.boxplot(column = 'minutes_asleep', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
# plt.xlabel('Difference in minutes in Bed between Weekend and Weekdays')
# plt.suptitle('')
# plt.title('');

# ax2 = f.add_subplot(122)
# new_df.boxplot(column = 'minutes_asleep', by = 'day', vert = False, widths = 0.4, ax=ax2)
# plt.xlabel('Difference in minutes in Bed each day')
# plt.suptitle('')
# plt.title('');

# plt.subplots_adjust(wspace = 1)


# # In[78]:


# f = plt.figure(figsize=(15,5))

# ax = f.add_subplot(121) 
# final_df.boxplot(column = 'Total_Steps', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
# plt.xlabel('Difference in number of steps between Weekend and Weekdays')
# plt.suptitle('')
# plt.title('');

# ax2 = f.add_subplot(122) 
# final_df.boxplot(column = 'Total_Steps', by = 'day', vert = False, widths = 0.4, ax=ax2)
# plt.xlabel('Difference in number of steps each day')
# plt.suptitle('')
# plt.title('');

# plt.subplots_adjust(wspace = 1)


# # ## Machine Learning

# # In[79]:


# from sklearn import tree
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score


# # In[80]:


# from sklearn.preprocessing import LabelEncoder

# number = LabelEncoder()
# new_df['day_num'] = number.fit_transform(new_df['day'].astype("str"))


# # In[81]:


# X = new_df[['day_num', 'Total_Steps', 'sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 
#             'very_active_minutes', 'overall_score', 'resting_heart_rate']]
# X.fillna(X.mean(), inplace=True)

# threshold = 2000

# Y = new_df['Total_Calories'] > threshold
# #Y = final_df['success']

# print('X shape: {}'.format(X.shape))
# print('Y shape: {}'.format(Y.shape))

# x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)
# print('X_train shape: {}. X_test shape: {}'.format(x_train.shape, x_test.shape))


# # In[82]:


# clf_final = tree.DecisionTreeClassifier(random_state=42)
# clf_final.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# fig, ax = plt.subplots(figsize=(12, 12))
# out = tree.plot_tree(clf_final, feature_names=list(x_train), filled = True)
# for o in out:
#     arrow = o.arrow_patch
#     if arrow is not None:
#         arrow.set_edgecolor('black')
#         arrow.set_linewidth(3)
# plt.show()


# # In[286]:


# from sklearn.tree.export import export_text
# tree_rules = export_text(clf_final, feature_names=list(x_train))


# # In[287]:


# tree_rules


# # In[282]:


# train_accuracy = clf.score(x_train, y_train) 
# val_accuracy = clf.score(x_test, y_test) 
# [train_accuracy,val_accuracy]


# # In[284]:


# accuracy = accuracy_score(y_test, y_pred)
# accuracy


# # In[251]:


# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(x_train, y_train)

# # Make predictions using the testing set
# lr_y_pred = regr.predict(x_test)
                        
# print("Mean squared error: %.2f"% mean_squared_error(y_test, lr_y_pred))
# #coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(regr.coef_))], axis = 1)
# #coefficients
# print("r2_error: %.2f"% r2_score(y_test, lr_y_pred))


# # In[ ]:




