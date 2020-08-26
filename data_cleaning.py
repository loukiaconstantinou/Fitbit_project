#import libraries
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
    parser.add_argument("--datapath", type=str, help="Input data path (combined_csv_files)", required=True)
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args

args = parse_args()
#get path
path = '{}/combined_csv_files_{}'.format(args.datapath, args.name)

#read sleep_score csv file
sleep_score = pd.read_csv('{}/sleep_score.csv'.format(path))

#read file function
def read_file(name):
    data_measured = pd.read_csv(path + '/combined_{}.csv'.format(name))
    return data_measured

#read files
steps = read_file('steps')
calories = read_file('calories')
distance = read_file('distance')
heart_rate = read_file('heart_rate')
resting_heart_rate = read_file('resting_heart_rate')
sedentary_minutes = read_file('sedentary_minutes')
very_active_minutes = read_file('very_active_minutes')
moderately_active_minutes = read_file('moderately_active_minutes')
lightly_active_minutes = read_file('lightly_active_minutes')

#create list of dataframes
data_frames = [sedentary_minutes, very_active_minutes, moderately_active_minutes, lightly_active_minutes]
#merge dataframes realting to activity in one dataframe
activity = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

#remove unecessary columns
activity = activity.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1)

#rename columns in activity dataframe
activity.columns = ['dateTime', 'sedentary_minutes', 'very_active_minutes', 'moderately_active_minutes', 'lightly_active_minutes']

#function to change type of dateTime column in a dataframe to datetime
def to_dt(name):
    name['dateTime'] = pd.to_datetime(name['dateTime'])
    return name

#change type of dateTime columns to datetime
activity = to_dt(activity)
calories = to_dt(calories)
distance = to_dt(distance)
resting_heart_rate = to_dt(resting_heart_rate)
steps = to_dt(steps)

#calculate total steps done each day
steps = steps.groupby(steps.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Steps')
#calculate total calories burned each day
calories = calories.groupby(calories.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Calories')
#calcualte total distance covered each day
distance = distance.groupby(distance.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Distance')

#change type of dateTime columns to datetime
steps = to_dt(steps)
calories = to_dt(calories)
distance = to_dt(distance)

#create dataframe by only getting sleeping score and date from dataframe
sleep_score = sleep_score[['timestamp', 'overall_score']]

#add empty columns
resting_heart_rate['dict_value'] = ""
resting_heart_rate['resting_heart_rate'] = ""
for i in range(len(resting_heart_rate)):
    #convert value column to dictionary
    resting_heart_rate['dict_value'][i] = ast.literal_eval(resting_heart_rate['value'][i])
    for key in resting_heart_rate['dict_value'][i]:
        #get the value from dictionary for each day (i.e. get the resting heart rate for each day)
        resting_heart_rate['resting_heart_rate'][i] = resting_heart_rate['dict_value'][i]['value']

#create dataframe by only getting resting heart rate and date columns from dataframe
resting_heart_rate = resting_heart_rate[['dateTime', 'resting_heart_rate']]
#change type of dateTime column to datetime
resting_heart_rate = to_dt(resting_heart_rate)

#change name of timestamp column to dateTime and to datetime type
sleep_score['dateTime'] = pd.to_datetime(sleep_score['timestamp'])
#remove time from dateTime column
sleep_score['dateTime'] = sleep_score['dateTime'].dt.
#remove timestamp column
sleep_score = sleep_score.drop(['timestamp'], axis = 1)
#change type of dateTime columns to datetime
sleep_score = to_dt(sleep_score)

#create list of dataframes
data_frames = [activity, steps, calories, distance, sleep_score, resting_heart_rate]
#merge the dataframes into one dataframe
final_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

#add column showing the day
final_df['day'] = final_df['dateTime'].dt.day_name()
#add column showing Y if day is in weekend (saturday or sunday) and N if day is not in weekend
final_df['is_weekend'] = np.where((final_df.day == 'Saturday') | (final_df.day == 'Sunday'), 'Y', 'N')
#add column showing Y if total number of steps during the day is greater than or equal to 10000 and N if total number of steps during the day is not greater than or equal to 10000
final_df['success_steps'] = np.where(final_df.Total_Steps >= 10000, 'Y', 'N')

#read file
sleep = read_file('sleep')

#add empty columns
sleep['dict_levels'] = ""
sleep['deep'] = ""
sleep['wake'] = ""
sleep['light'] = ""
sleep['rem'] = ""
sleep['restless'] = ""
sleep['awake'] = ""
sleep['asleep'] = ""
for i in range(len(sleep)):
    #convert levels column to dictionary
    sleep['dict_levels'][i] = ast.literal_eval(sleep['levels'][i])
    for key in sleep['dict_levels'][i]['summary']:
        #get the value for each sleeping state from dictionary for each day (i.e. get the time in deep, wake, light, rem, restless, awake, asleep for each day)
        sleep[key][i] = sleep['dict_levels'][i]['summary'][key]['minutes']

#create dataframe by only getting time in different sleeping states and date column
sleep = sleep[['dateOfSleep', 'deep', 'wake', 'light', 'rem', 'restless', 'awake', 'asleep']]

#rename date column so that it is consistent with the other dataframes
sleep = sleep.rename(columns={"dateOfSleep": "dateTime"})

#change type of dateTime columns to datetime
sleep = to_dt(sleep)

#create list of dataframes
data_frames = [final_df, sleep]
#merge the dataframes into one dataframe (i.e. this will add the times in the different sleeping states to the previous merged dataframe)
final_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

#remove rows that have at least one na value in them
new_df = final_df.dropna()

#save the new dataframe as a csv
new_df.to_csv(r"/{}_final_df.csv".format(path), index = False)



