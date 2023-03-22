import pandas as pd 
import numpy as np 
import math
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

student_nr = 's4917340'
root_path = os.path.join(sys.path[0])
data_path = os.path.join(root_path, 'data')
plot_path = os.path.join(root_path, 'plot')
# pd.set_option('display.max_rows', None)

def data_preparation(machine_data : pd.DataFrame):
    machine_data['Censored'] = machine_data['Event'].map({'failure': 'No', 'PM':'Yes'})
    machine_data['Duration'] = machine_data['Time'].diff(periods=1).fillna(machine_data['Time'].iloc[0])
    machine_data = machine_data.sort_values(by='Duration', ascending=True)
    # source: https://stackoverflow.com/questions/67845362/sort-pandas-df-subset-of-rows-within-a-group-by-specific-column 
    # source: https://sparkbyexamples.com/pandas/pandas-groupby-sort-within-groups/
    machine_data = machine_data.groupby('Duration', group_keys=True).apply(lambda x: x.sort_values(by=['Event'], ascending=False))
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

# Mean time between failures (Weibull)
def meantimebetweenfailure_weibull(lamb_val, kap_val):
    return lamb_val * math.gamma(1+(1/kap_val))

def plot_kaplanmeier_estimation(data, machine_name):
    plt.title(f'Kaplan-Meier estimator for machine {machine_name}')
    sns.lineplot(x="Duration", y="Reliability", data=data)
    plt.savefig(os.path.join(plot_path, f'Kaplan-Meier-results-machine-{machine_name}.png'))

#Weibull distribution fitting
def fit_weibull_distribution(prepared_data : pd.DataFrame):
    # 1. Create a variable with the search ranges for lambda and kappa
    l_range = np.linspace(start=1, stop=35, num=10)
    k_range = np.linspace(start=0.1, stop=3.5, num=10)

    # 2. Create a dataframe which will contain your likelihood data
    #source: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays 
    pairs = np.array(list(itertools.product(l_range, k_range)))
    df_weibull = pd.DataFrame(columns=['Lambda', 'Kappa'], data=pairs)

    # 3. Create  log-likelihood  data  for  each  event  in  your  dataset.
    for observation_index, observation_row in prepared_data.iterrows():
        column_name = 'Observation ' + str(observation_index)
        for row_index in range(len(df_weibull)):
            l = df_weibull.loc[row_index, 'Lambda']
            k = df_weibull.loc[row_index, 'Kappa']

            duration = observation_row['Duration']
            censored = observation_row['Censored']

            if censored == 'No':
                #log (f(x)) density function : likelihood of failure at time t (failure) 
                fx = (k / l) * (duration / l) ** (k-1) * math.exp(-(duration/l)**k)
                log_likelihood = np.log(fx)
            else:
                # log (R(x)) = log (1-F(x)) probability of survival before time t (PM)
                Rx = math.exp(-(duration/l)**k)
                log_likelihood = np.log(Rx)
            df_weibull.loc[row_index, column_name] = log_likelihood

    # 4. Calculate the sum of the log-likelihoods. 
    # get all the column names starting with Observation
    observation_columns = df_weibull.columns[df_weibull.columns.str.contains(pat='Observation')].tolist()
    # sum the loglikelihood of each observation per row
    df_weibull['Loglikelihood_sum'] = df_weibull[observation_columns].sum(axis=1)
    print(df_weibull)
    # get the optimal Lambda and Kappa based on the max Loglikelihood sum
    max_loglikelihood_sum = df_weibull['Loglikelihood_sum'].max() 
    best_l_k = df_weibull[df_weibull['Loglikelihood_sum'] == max_loglikelihood_sum][['Lambda', 'Kappa']]
    return best_l_k['Lambda'].values[0], best_l_k['Kappa'].values[0]

def create_weibull_curve_data(prepared_data : pd.DataFrame, lamb_val, kap_val):
    # Create an new dataframe with t durations and reliability R_t
    weibull_data = pd.DataFrame(columns=['t', 'R_t'])
    # Create an range of durations from the 0 to length of the prepared data 
    weibull_data['t'] = np.arange(0, prepared_data['Duration'].max(), 0.01)
    # Reliability distribution function. The opposite of F(t), the probability of survival until t
    weibull_data['R_t'] = np.exp(-(weibull_data['t']/lamb_val)**kap_val) 
    return weibull_data

def visualization(KM_data : pd.DataFrame, weibull_data : pd.DataFrame, machine_name):
    fig, ax = plt.subplots()
    ax.set_title(f'Visualization of reliability functions of machine {machine_name}')
    ax.step(KM_data['Duration'], KM_data['Reliability'], where="post", label='Kaplan-Meier')
    ax.plot(weibull_data['t'], weibull_data['R_t'], label='Weibull')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join(plot_path, f'{student_nr}-Machine-{machine_name}-Reliability.png'))
    plt.close()

# Create a plot of the cost rates
def plot_cost_rates(maintenance_data : pd.DataFrame, machine_name):
    fig, ax = plt.subplots()
    x_min, x_max = 1, round(maintenance_data['t'].max())

    ax.set_title(f'Maintenance age impact on cost for machine {machine_name}')
    ax.plot(maintenance_data['t'], maintenance_data['cost_rate'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Cost')
    plt.savefig(os.path.join(plot_path, f'{student_nr}-Machine-{machine_name}-Costs.png'))

def create_cost_data(prepared_data : pd.DataFrame, l, k, PM_cost, CM_cost, machine_name):
    # 1. Define a range of maintenance ages 
    maintenance_data = pd.DataFrame()
    maintenance_data['t'] = np.arange(0, prepared_data['Duration'].max(), 0.01)
    # 2. Calculate the values F(t) and R(t) for each maintenance age
    maintenance_data['F_t'] = 1 - np.exp(-(maintenance_data['t']/l)**k) 
    maintenance_data['R_t'] = np.exp(-(maintenance_data['t']/l)**k) 
    # 3. Calculate the mean cost per cycle (MCPC) for each maintenance age 
    maintenance_data['MCPC'] = CM_cost * maintenance_data['F_t'] + PM_cost * maintenance_data['R_t']
    # 4. Calculate the mean cycle length (MCL) by approximating the area under curve using Riemann sum
    delta = 0.01
    t = maintenance_data['t'].values
    R_t = maintenance_data['R_t'].values
    riemann_sum = 0
    for i in range(len(t)):
        riemann_sum += R_t[i] * delta

    # 5. Calculate the cost rate for each maintenance age
    maintenance_data['cost_rate'] = maintenance_data['MCPC'] / riemann_sum
    print(maintenance_data)

    # 6. Create a plot of cost rates
    plot_cost_rates(maintenance_data, machine_name)

    # 7. determine the optimal maintenance age and the corresponding cost rate 
    optimal = maintenance_data[maintenance_data['cost_rate'] == maintenance_data['cost_rate'].min()]
    return optimal['t'].values[0], optimal['cost_rate'].values[0]

def run_analysis():
    machine_name = 1
    machine_data = pd.read_csv(os.path.join(data_path, f'{student_nr}-Machine-{machine_name}.csv'))
    prepared_data = data_preparation(machine_data)
    KM_data = create_kaplanmeier_data(prepared_data)
    plot_kaplanmeier_estimation(KM_data[['Duration', 'Reliability']], machine_name)
    MTBF_KM = meantimebetweenfailures_KM(KM_data)

    # Weibull fitting
    print(prepared_data)
    lamb_val, kap_val = fit_weibull_distribution(prepared_data)
    print(f'\nBest Lambda & Kappa values for machine {machine_name} are {lamb_val} and {kap_val}')

    # MTBF 
    MTBF_weibull = meantimebetweenfailure_weibull(lamb_val, kap_val)
    print('The MTBF-Kaplan Meier is:', MTBF_KM)
    print('The MTBF-Weibull is:', MTBF_weibull)

    # Weibull data
    weibull_data = create_weibull_curve_data(prepared_data, lamb_val, kap_val)

    visualization(KM_data, weibull_data, machine_name)

    # Age-based maintenance
    PM_cost = 5
    CM_cost = 20
    best_age, best_cost_rate = create_cost_data(prepared_data, lamb_val, kap_val, PM_cost, CM_cost, machine_name)
    print(best_age, best_cost_rate)
    return

run_analysis()





