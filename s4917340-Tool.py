import pandas as pd 
import numpy as np 
import math
import os
import sys

student_nr = 's4917340'
path = os.path.join(sys.path[0], 'data')

def check_for_duplicates_PM():
    return # what do I have to do here?


def data_preparation(machine_data : pd.DataFrame):
    machine_data['Duration'] = machine_data['Time'].diff(periods=1).fillna(0)
    machine_data = machine_data.sort_values(by='Duration', ascending=True)
    machine_data = machine_data.drop_duplicates(subset=['Duration']) #TODO: I have to check if I just remove the dups 
    machine_data = machine_data.reset_index(drop=True)
    return machine_data

def update_probabilities(row): #TODO: this is not correct yet 
    print(row)
    if row['Event'] == 'PM':
        remaining_rows = row.iloc[row.name+1:] # get all rows under the current row
        num_remaining_rows = len(remaining_rows) # number of remaining rows
        prob_to_distribute = row['Probability'] # get the current probability of the current row
        prob_per_row = prob_to_distribute / num_remaining_rows # probablity value to divide over the remaining rows
        remaining_rows.loc[:, 'probability'] += prob_per_row # add proability to all remaining rows
        row['probability'] = 0 # finally, set the current proability of current row to zero
    return row

def create_kaplanmeier_data(prepared_data : pd.DataFrame):
    prepared_data['Probability'] = 1/len(prepared_data)
    prepared_data = prepared_data.apply(update_probabilities, axis=1)
    return prepared_data

def run_analysis():
    machine_name = 1
    machine_data = pd.read_csv(os.path.join(path, f'{student_nr}-Machine-{machine_name}.csv'))
    prepared_data = data_preparation(machine_data)
    KM_data = create_kaplanmeier_data(prepared_data)
    return KM_data


result = run_analysis()
print(result)

