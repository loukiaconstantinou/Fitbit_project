#import libraries
import pandas as pd
import os
from functools import reduce
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#parser function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Input data path (dataframe_csv)", default= os.getcwd())
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args

args = parse_args()

if not os.path.isdir("plots_{}".format(args.name)):
    os.makedirs("plots_{}".format(args.name))

path = 'plots_{}'.format(args.name)
new_df = pd.read_csv('combined_csv_files_{}_final_df.csv'.format(args.name), header = 0)

#convert dateTime column to datetime type
new_df['dateTime'] = pd.to_datetime(new_df['dateTime'])
#print(new_df.dtypes)
#print(type(new_df))

# new_df.columns = new_df.columns.to_series().apply(lambda x: x.strip())

## EDA

#Correlation Matrix
#dataframe columns that will be included in the correlation matrix
corrdf_calories = new_df[['Total_Calories','Total_Steps', 'is_weekend', 'sedentary_minutes', 'very_active_minutes', 
                            'moderately_active_minutes', 'lightly_active_minutes', 'resting_heart_rate', 'overall_score']]

#correlation matrix function
def CorrMtx(df, dropDuplicates = True):
    #make dataframe a correlation matrix
    df = df.corr()

    #exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    #set background color
    sns.set_style(style = 'white')

    #set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    #add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
  
    #draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)  
        f.savefig('{}svm_conf.png'.format(path), dpi=400) #save the figure
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        f.savefig('{}/svm_conf.png'.format(path), dpi=400) #save the figure

#plot correlation matrix
CorrMtx(corrdf_calories, dropDuplicates = False)

#use pairplot function to plot pairplot of the same features
sns.pairplot(corrdf_calories.dropna(), kind="scatter", markers="+", plot_kws=dict(s=50, edgecolor="b", linewidth=1))
plt.savefig('{}/pairplot.png'.format(path), dpi=400) #save the figure

#Step Analysis
#set up matplot figure
f = plt.figure(figsize=(15,5))

#first figure
ax = f.add_subplot(121) 
#plot boxplot of total steps completed during weekend and weekdays
new_df.boxplot(column = 'Total_Steps', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
plt.xlabel('Difference in number of steps between Weekend and Weekdays') #remame the x-axis title
plt.ylabel('Is Weekend') #rename the y-axis title
plt.suptitle('')
plt.title('');

#second figure
ax2 = f.add_subplot(122) 
#plot boxplot of total steps completed for each day
new_df.boxplot(column = 'Total_Steps', by = 'day', vert = False, widths = 0.4, ax=ax2)
plt.xlabel('Difference in number of steps each day') #rename the x-axis title
plt.ylabel('Day') #rename the y-axis title
plt.suptitle('')
plt.title('');

#adjust whitespace between boxplots
plt.subplots_adjust(wspace = 1)
plt.savefig('{}/box_plots_steps.png'.format(path), dpi = 400) #save the figure

#Day Analysis
#bar plots
#set up matplot figure
fig = plt.figure(figsize = (20,6))

#first figure
ax = plt.subplot(141)  
#group by day and find the average number of steps completed in one day & plot the bar chart
new_df.groupby('day').Total_Steps.mean().plot.bar()
plt.title('Day of Week vs. Steps', fontsize=15) #rename the title fo the figure
plt.xlabel('Day of Week', fontsize=14) #rename the x-axis title
plt.ylabel('Steps', fontsize=14) #rename the y-axis title
#include horizontal dashed line for 8000 step goal
ax.axhline(8000, color="orangered", linestyle='--')
#include horizontal dashed line for 10000 step goal
ax.axhline(10000, color="orange", linestyle='--')

#second figure
ax2 = fig.add_subplot(142)
#group by day and find the average calories burned in one day & plot the bar chart
new_df.groupby('day').Total_Calories.mean().plot.bar()
plt.title('Day of Week vs. Calories Burned', fontsize=15) #rename the title fo the figure
plt.xlabel('Day of Week', fontsize=14) #rename the x-axis title
plt.ylabel('Calories Burned', fontsize=14) #rename the y-axis title

#third figure
ax3 = fig.add_subplot(143)
#group by day and find the average time the user is very active in one day
new_df.groupby('day').very_active_minutes.mean().plot.bar()
plt.title('Day of Week vs. Minutes Very Active', fontsize=15) #rename the title fo the figure
plt.xlabel('Day of Week', fontsize=14) #rename the x-axis title
plt.ylabel('Minutes Very Active', fontsize=14) #rename the y-axis title

#fourth figure
ax4 = fig.add_subplot(144)
#group by day and find the average time the user is sedentary in one day & plot the bar chart
new_df.groupby('day').sedentary_minutes.mean().plot.bar()
plt.title('Day of Week vs. Minutes Sedentary', fontsize=15) #rename the title fo the figure
plt.xlabel('Day of Week', fontsize=14) #rename the x-axis title
plt.ylabel('Minutes Sedentary', fontsize=14) #rename the y-axis title
fig.savefig('{}/bar_days.png'.format(path), dpi = 400) #save the figure

#line graph
#set up matplot figure
fig = plt.figure(figsize = (10,8))
#group by day and find the average resting heart rate in one day & plot the line graph
new_df.groupby('day').resting_heart_rate.mean().plot.line()
plt.title('Day of Week vs. Resting Heart Rate', fontsize=15) #rename the title fo the figure
plt.xlabel('Day of Week', fontsize=14) #rename the x-axis title
plt.ylabel('Resting Heart Rate', fontsize=14) #rename the y-axis title
fig.savefig('{}/line_days.png'.format(path), dpi = 400) #save the figure

#Sleep Analysis

#total minutes asleep
new_df['minutes_asleep'] = new_df['deep'] + new_df['wake'] + new_df['light'] + new_df['rem']
new_df = new_df.replace("", 0)
new_df = new_df.reset_index()

#if the user did not sleep for a significant amount of time, then the sleep cycle is not analysed (not split in different stages) and will only include the time the user was generally asleep
#so total minutes asleep will be equal to the general time the user was asleep
for i in range(len(new_df)):
    if new_df['minutes_asleep'][i] == 0:
        new_df['minutes_asleep'][i] = new_df['asleep'][i]
        
#boxplots
#set up matplot figure
f = plt.figure(figsize=(15,5))

#first figure
ax = f.add_subplot(121)
#plot boxplot of minutes asleep during weekend and weekdays
new_df.boxplot(column = 'minutes_asleep', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
plt.xlabel('Difference in minutes in Bed between Weekend and Weekdays') #remame the x-axis title
plt.ylabel('Is Weekend') #rename the y-axis title
plt.suptitle('')
plt.title('');

#second figure
ax2 = f.add_subplot(122)
#plot boxplot of minutes asleep for each day
new_df.boxplot(column = 'minutes_asleep', by = 'day', vert = False, widths = 0.4, ax=ax2)
plt.xlabel('Difference in minutes in Bed each day') #rename the x-axis title
plt.ylabel('Day') #rename the y-axis title
plt.suptitle('')
plt.title('');

#adjust whitespace between boxplots
plt.subplots_adjust(wspace = 1)
plt.savefig('{}/box_plots_sleep.png'.format(path), dpi = 400) #save the figure

#pie chart
#get only data that include information for all stages in sleep cycle
row = ((new_df.loc[new_df['deep'] == 0]))

sleep_perc_df = new_df[~new_df.index.isin(row.index)]

#add column showing proportion of time the user was in a sleeping stage to total time the user was asleep
sleep_perc_df['deep_perc'] = sleep_perc_df['deep']/sleep_perc_df['minutes_asleep']
sleep_perc_df['wake_perc'] = sleep_perc_df['wake']/sleep_perc_df['minutes_asleep']
sleep_perc_df['light_perc'] = sleep_perc_df['light']/sleep_perc_df['minutes_asleep']
sleep_perc_df['rem_perc'] = sleep_perc_df['rem']/sleep_perc_df['minutes_asleep']

#find mean of each new column
avg_perc_sleep = sleep_perc_df[['deep_perc', 'wake_perc', 'light_perc', 'rem_perc']].mean()

#set up matplot figure
fig = plt.figure(figsize = (6,6))
#set labels
labels=['Deep sleep', 'Awake', 'Light sleep', 'REM sleep']
#plot the pie chart
plt.pie(avg_perc_sleep, colors = ['darkturquoise', 'salmon', 'lightskyblue', 'yellowgreen'], autopct='%1.1f%%', labels=labels, textprops=dict(color="w"))

plt.title('Average of types of sleep', fontsize=14) #rename the title of the figure
plt.legend() #add legend
fig.savefig('{}/pie_sleep.png'.format(path), dpi = 400) #save the figure

#bar chart
#group by day and find the average time the user is in each each stage in the sleep cycle & plot the bar chart
sleep_perc_df.groupby('day').mean()[["deep", "wake", "light", "rem"]].plot(kind='bar')
plt.savefig('{}/days_sleep.png'.format(path), dpi = 400) #save the figure

#Resting Heart Rate Analysis
#group by week and find the average time the user was very active during each week
week_very_active = new_df.groupby([new_df['dateTime'].dt.strftime('%W')])['very_active_minutes'].mean()

#change type of resting_heart_rate column to numeric
new_df['resting_heart_rate'] = new_df['resting_heart_rate'].apply(pd.to_numeric, downcast='float', errors='coerce')

#group by week and find the resting heart rate during each week
week_rest_heart_rate = new_df.groupby([new_df['dateTime'].dt.strftime('%W')])['resting_heart_rate'].mean()

#create list of dataframes
data_frames = [week_very_active, week_rest_heart_rate]
#merge weekly dataframes into one
week_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

#set up matplot figure
fig = plt.figure(figsize = (10, 8))
#plot scatter plot of very active minutes against resting heart rate
plt.scatter(week_df['very_active_minutes'], week_df['resting_heart_rate'])
plt.xlabel('very_active_minutes') #rename the x-axis title
plt.ylabel('resting_heart_rate') #rename the y-axis title
fig.savefig('{}/v_act_rest.png'.format(path)) #save the figure
