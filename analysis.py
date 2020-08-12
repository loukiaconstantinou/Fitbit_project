import pandas as pd
import os
from functools import reduce
import ast
import numpy as np

#parser function to get datapath and name of person to make analysis on
def parse_args():
    import argparse
    #define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Input data path (dataframe_csv)", default= os.getcwd())
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args
#call the parser function
args = parse_args()

if not os.path.isdir("plots_{}".format(args.name)):
    os.makedirs("plots_{}".format(args.name))

path = 'plots_{}'.format(args.name)
new_df = pd.read_csv('combined_csv_files_{}_final_df.csv'.format(args.name), header = 0)

new_df['dateTime'] = pd.to_datetime(new_df['dateTime'])
print(new_df.dtypes)
print(type(new_df))

# new_df.columns = new_df.columns.to_series().apply(lambda x: x.strip())

## EDA

corrdf_calories = new_df[['Total_Calories','Total_Steps', 'is_weekend', 'sedentary_minutes', 'very_active_minutes', 
                            'moderately_active_minutes', 'lightly_active_minutes', 'resting_heart_rate', 'overall_score']]

import seaborn as sns
import matplotlib.pyplot as plt
def CorrMtx(df, dropDuplicates = True):

    # Your dataset is already a correlation matrix.
    # If you have a dateset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)  
        f.savefig('{}svm_conf.png'.format(path), dpi=400)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        f.savefig('{}/svm_conf.png'.format(path), dpi=400)


CorrMtx(corrdf_calories, dropDuplicates = False)


sns.pairplot(corrdf_calories.dropna(), kind="scatter", markers="+", plot_kws=dict(s=50, edgecolor="b", linewidth=1))
plt.savefig('{}/pairplot.png'.format(path), dpi=400)


week_sed = new_df.groupby([new_df['dateTime'].dt.strftime('%W')])['sedentary_minutes'].mean()


new_df['resting_heart_rate'] = new_df['resting_heart_rate'].apply(pd.to_numeric, downcast='float', errors='coerce')


week_rest_heart_rate = new_df.groupby([new_df['dateTime'].dt.strftime('%W')])['resting_heart_rate'].mean()



data_frames = [week_sed, week_rest_heart_rate]
week_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

fig = plt.figure(figsize = (10, 8))
plt.scatter(week_df['sedentary_minutes'], week_df['resting_heart_rate'])
plt.xlabel('sedentary_minutes')
plt.ylabel('resting_heart_rate')
fig.savefig('{}/sed_rest.png'.format(path), dpi = 400)


week_very_active = new_df.groupby([new_df['dateTime'].dt.strftime('%W')])['very_active_minutes'].mean()


data_frames = [week_very_active, week_rest_heart_rate]
week_df2 = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

fig = plt.figure(figsize = (10, 8))
plt.scatter(week_df2['very_active_minutes'], week_df2['resting_heart_rate'])
plt.xlabel('very_active_minutes')
plt.ylabel('resting_heart_rate')
fig.savefig('{}/v_act_rest.png'.format(path))

fig = plt.figure(figsize = (20,6))

ax = plt.subplot(141)  
new_df.groupby('day').Total_Steps.mean().plot.bar()
plt.title('Day of Week vs. Steps', fontsize=15)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Steps', fontsize=14)
ax.axhline(8000, color="orangered", linestyle='--')
ax.axhline(10000, color="orange", linestyle='--')

ax2 = fig.add_subplot(142)
new_df.groupby('day').Total_Calories.mean().plot.bar()
plt.title('Day of Week vs. Calories Burned', fontsize=15)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Calories Burned', fontsize=14)

ax3 = fig.add_subplot(143)
new_df.groupby('day').very_active_minutes.mean().plot.bar()
plt.title('Day of Week vs. Minutes Very Active', fontsize=15)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Minutes Very Active', fontsize=14)

ax4 = fig.add_subplot(144)
new_df.groupby('day').sedentary_minutes.mean().plot.bar()
plt.title('Day of Week vs. Minutes Sedentary', fontsize=15)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Minutes Sedentary', fontsize=14)
fig.savefig('{}/bar_days.png'.format(path), dpi = 400)

fig = plt.figure(figsize = (10,8))
new_df.groupby('day').resting_heart_rate.mean().plot.line()
plt.title('Day of Week vs. Resting Heart Rate', fontsize=15)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Resting Heart Rate', fontsize=14)
fig.savefig('{}/line_days.png'.format(path), dpi = 400)



new_df['minutes_asleep'] = new_df['deep'] + new_df['wake'] + new_df['light'] + new_df['rem']
new_df = new_df.replace("", 0)
new_df = new_df.reset_index()


for i in range(len(new_df)):
    if new_df['minutes_asleep'][i] == 0:
        new_df['minutes_asleep'][i] = new_df['asleep'][i]


row = ((new_df.loc[new_df['deep'] == 0]))

sleep_perc_df = new_df[~new_df.index.isin(row.index)]


sleep_perc_df['deep_perc'] = sleep_perc_df['deep']/sleep_perc_df['minutes_asleep']
sleep_perc_df['wake_perc'] = sleep_perc_df['wake']/sleep_perc_df['minutes_asleep']
sleep_perc_df['light_perc'] = sleep_perc_df['light']/sleep_perc_df['minutes_asleep']
sleep_perc_df['rem_perc'] = sleep_perc_df['rem']/sleep_perc_df['minutes_asleep']


avg_perc_sleep = sleep_perc_df[['deep_perc', 'wake_perc', 'light_perc', 'rem_perc']].mean()

fig = plt.figure(figsize = (6,6))
labels=['Deep sleep', 'Awake', 'Light sleep', 'REM sleep']
plt.pie(avg_perc_sleep, colors = ['darkturquoise', 'salmon', 'lightskyblue', 'yellowgreen'], autopct='%1.1f%%', labels=labels, textprops=dict(color="w"))

plt.title('Average of types of sleep', fontsize=14)
plt.legend()
fig.savefig('{}/pie_sleep.png'.format(path), dpi = 400)


sleep_perc_df.groupby('day').mean()[["deep", "wake", "light", "rem"]].plot(kind='bar')
plt.savefig('{}/days_sleep.png'.format(path), dpi = 400)


f = plt.figure(figsize=(15,5))

ax = f.add_subplot(121)
new_df.boxplot(column = 'minutes_asleep', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
plt.xlabel('Difference in minutes in Bed between Weekend and Weekdays')
plt.suptitle('')
plt.title('');

ax2 = f.add_subplot(122)
new_df.boxplot(column = 'minutes_asleep', by = 'day', vert = False, widths = 0.4, ax=ax2)
plt.xlabel('Difference in minutes in Bed each day')
plt.suptitle('')
plt.title('');

plt.subplots_adjust(wspace = 1)
plt.savefig('{}/box_plots_sleep.png'.format(path), dpi = 400)


f = plt.figure(figsize=(15,5))

ax = f.add_subplot(121) 
new_df.boxplot(column = 'Total_Steps', by = 'is_weekend', vert = False, widths = 0.4, ax=ax)
plt.xlabel('Difference in number of steps between Weekend and Weekdays')
plt.suptitle('')
plt.title('');

ax2 = f.add_subplot(122) 
new_df.boxplot(column = 'Total_Steps', by = 'day', vert = False, widths = 0.4, ax=ax2)
plt.xlabel('Difference in number of steps each day')
plt.suptitle('')
plt.title('');

plt.subplots_adjust(wspace = 1)
plt.savefig('{}/box_plots_steps.png'.format(path), dpi = 400)
