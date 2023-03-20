import pandas as pd 
import numpy as np 
import math
import os
import sys

student_nr = 's4917340'
path = os.path.join(sys.path[0], 'data')
pd.set_option('display.max_rows', None)

def data_preparation(machine_data : pd.DataFrame):
    machine_data['Censored'] = machine_data['Event'].map({'failure': 'No', 'PM':'Yes'})
    machine_data['Duration'] = machine_data['Time'].diff(periods=1).fillna(0)
    machine_data = machine_data.sort_values(by='Duration', ascending=True)

    # source: https://stackoverflow.com/questions/67845362/sort-pandas-df-subset-of-rows-within-a-group-by-specific-column 
    # source: https://sparkbyexamples.com/pandas/pandas-groupby-sort-within-groups/
    machine_data = machine_data.groupby('Duration').apply(lambda x: x.sort_values(by=['Event'], ascending=False))
    machine_data = machine_data.reset_index(drop=True)

    return machine_data

# Kaplan-Meier estimator for updating the probablities based on observed events 
def update_probabilities(prepared_data : pd.DataFrame):
    for index, row in prepared_data.iterrows():
        if row['Event'] == 'PM':
            remaining_rows = prepared_data.iloc[index+1:]
            num_remaining_rows = len(remaining_rows)
            prob_to_distribute = row['Probability']
            prob_to_distribute_per_row = prob_to_distribute / num_remaining_rows
            remaining_rows.loc[:, 'Probability'] += prob_to_distribute_per_row
            prepared_data.at[index, 'Probability'] = 0
    
    return prepared_data

def create_kaplanmeier_data(prepared_data : pd.DataFrame):
    # 1. Add a column named probability to the dataframe
    prepared_data['Probability'] = 1/len(prepared_data)
    
    # 2. Update proabilities based on the observed events 
    prepared_data = update_probabilities(prepared_data)
    
    # 3. Merge duplicated, failure (censored == No) durations 
    # group the dataframe with duplicates by duration and aggrated sum on proability
    grouped = prepared_data.groupby('Duration').agg({'Probability': 'sum'}).reset_index()

    prepared_data['summed_prob'] = prepared_data['Duration'].map(grouped.set_index('Duration')['Probability'])
        
    mask = (prepared_data['Event'] == 'failure') & (prepared_data.duplicated(subset='Duration', keep='first'))
    prepared_data.loc[mask, 'Probability'] = prepared_data.loc[mask, 'summed_prob']
    prepared_data.loc[~mask, 'Probability'] = 0
    
    return prepared_data

def run_analysis():
    machine_name = 'test'
    machine_data = pd.read_csv(os.path.join(path, f'{student_nr}-Machine-{machine_name}.csv'))
    prepared_data = data_preparation(machine_data)
    KM_data = create_kaplanmeier_data(prepared_data)
    return KM_data


result = run_analysis()