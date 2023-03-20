import pandas as pd 
import numpy as np 
import math
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

student_nr = 's4917340'
root_path = os.path.join(sys.path[0])
data_path = os.path.join(root_path, 'data')
plot_path = os.path.join(root_path, 'plot')
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

# 2. Kaplan-Meier estimator for updating the probablities based on observed events 
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
    # group the dataframe with duplicates by duration and aggrated sum on proability column
    grouped = prepared_data.groupby('Duration').agg({'Probability': 'sum'}).reset_index()
    prepared_data['summed_prob'] = prepared_data['Duration'].map(grouped.set_index('Duration')['Probability'])
    mask = (prepared_data.duplicated('Duration', keep=False)) & (prepared_data['Event'] == 'failure') #mark all duplicates
    prepared_data.loc[mask, 'Probability'] = prepared_data.loc[mask, 'summed_prob']
    mask2 = (prepared_data.duplicated('Duration')) & (prepared_data['Event'] == 'failure') # mark all dups besides the first one
    prepared_data.loc[mask2, 'Probability'] = 0
    prepared_data = prepared_data.drop('summed_prob', axis=1)

    # 4. Calculate the reliability function for each duration 
    reliability = 1
    for index, row in prepared_data.iterrows():
        if row['Event'] == 'failure':
            reliability -= row['Probability']
            prepared_data.at[index, 'Reliability'] = reliability
        else:
            prepared_data.at[index, 'Reliability'] = reliability
    return prepared_data

# Mean time between failures (Kaplan-Meier)
def meantimebetweenfailures_KM(KM_data : pd.DataFrame):
    KM_data['MTBF'] = KM_data['Duration'] * KM_data['Probability']
    return KM_data['MTBF'].sum()

def plot_kaplanmeier_estimation(data, machine_name):
    plt.title(f'Kaplan-Meier estimator for machine {machine_name}')
    sns.lineplot(x="Duration", y="Reliability", data=data)
    plt.savefig(os.path.join(plot_path, f'Kaplan-Meier-results-machine-{machine_name}.png'))

def run_analysis():
    machine_name = 'test'
    machine_data = pd.read_csv(os.path.join(data_path, f'{student_nr}-Machine-{machine_name}.csv'))
    prepared_data = data_preparation(machine_data)
    KM_data = create_kaplanmeier_data(prepared_data)
    MTBF_KM = meantimebetweenfailures_KM(KM_data)
    print('The MTBF-KaplanMeier is: ', MTBF_KM)
    return KM_data

result = run_analysis()
print(result)





