import pandas as pd
import os
from functools import reduce
import ast
import numpy as np

#parser function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Input data path (combined_csv_files)", required=True)
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args

args = parse_args()
path = '{}/combined_csv_files_{}'.format(args.datapath, args.name)
print(path)


sleep_score = pd.read_csv('{}/sleep_score.csv'.format(path))

def read_file(name):
    data_measured = pd.read_csv(path + '/combined_{}.csv'.format(name))
    return data_measured

steps = read_file('steps')
calories = read_file('calories')
distance = read_file('distance')
heart_rate = read_file('heart_rate')
resting_heart_rate = read_file('resting_heart_rate')
sedentary_minutes = read_file('sedentary_minutes')
very_active_minutes = read_file('very_active_minutes')
moderately_active_minutes = read_file('moderately_active_minutes')
lightly_active_minutes = read_file('lightly_active_minutes')

data_frames = [sedentary_minutes, very_active_minutes, moderately_active_minutes, lightly_active_minutes]
activity = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)

activity = activity.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1)

activity.columns = ['dateTime', 'sedentary_minutes', 'very_active_minutes', 'moderately_active_minutes', 'lightly_active_minutes']

def to_dt(name):
    name['dateTime'] = pd.to_datetime(name['dateTime'])
    return name

activity = to_dt(activity)
calories = to_dt(calories)
distance = to_do(distance)
resting_heart_rate = to_do(resting_heart_rate)


steps = steps.groupby(steps.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Steps')
calories = calories.groupby(calories.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Calories')
distance = distance.groupby(distance.dateTime.dt.date, sort=False)['value'].sum().reset_index(name ='Total_Distance')

steps = to_dt(steps)
calories = to_dt(calories)
distance = to_do(distance)

sleep_score = sleep_score[['timestamp', 'overall_score']]

resting_heart_rate['dict_value'] = ""
resting_heart_rate['resting_heart_rate'] = ""
for i in range(len(resting_heart_rate)):
    resting_heart_rate['dict_value'][i] = ast.literal_eval(resting_heart_rate['value'][i])
    for key in resting_heart_rate['dict_value'][i]:
        resting_heart_rate['resting_heart_rate'][i] = resting_heart_rate['dict_value'][i]['value']


resting_heart_rate = resting_heart_rate[['dateTime', 'resting_heart_rate']]
resting_heart_rate['dateTime'] = pd.to_datetime(resting_heart_rate['dateTime'])

sleep_score['dateTime'] = pd.to_datetime(sleep_score['timestamp'])
sleep_score['dateTime'] = sleep_score['dateTime'].dt.date
sleep_score = sleep_score.drop(['timestamp'], axis = 1)
sleep_score = to_dt(sleep_score)

data_frames = [activity, steps, calories, distance, sleep_score, resting_heart_rate]
for i in enumerate(data_frames):
    print(i[1].dtypes)

final_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)


final_df['day'] = final_df['dateTime'].dt.day_name()
final_df['is_weekend'] = np.where((final_df.day == 'Saturday') | (final_df.day == 'Sunday'), 'Y', 'N')
final_df['success_steps'] = np.where(final_df.Total_Steps >= 10000, 'Y', 'N')


sleep = read_file('sleep')

sleep['dict_levels'] = ""
sleep['deep'] = ""
sleep['wake'] = ""
sleep['light'] = ""
sleep['rem'] = ""
sleep['restless'] = ""
sleep['awake'] = ""
sleep['asleep'] = ""
for i in range(len(sleep)):
    sleep['dict_levels'][i] = ast.literal_eval(sleep['levels'][i])
    for key in sleep['dict_levels'][i]['summary']:
        sleep[key][i] = sleep['dict_levels'][i]['summary'][key]['minutes']


sleep = sleep[['dateOfSleep', 'deep', 'wake', 'light', 'rem', 'restless', 'awake', 'asleep']]

sleep = sleep.rename(columns={"dateOfSleep": "dateTime"})

sleep['dateTime'] = pd.to_datetime(sleep['dateTime'])

data_frames = [final_df, sleep]
final_df = reduce(lambda  left,right: pd.merge(left,right,on=['dateTime'],
                                            how='outer'), data_frames)


new_df = final_df.dropna()
new_df.to_csv(r"/{}_final_df3.csv".format(path), index = False)


