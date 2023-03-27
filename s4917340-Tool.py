import pandas as pd 
import numpy as np 
import math
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore') 

student_nr = 's4917340'
root_path = os.path.join(sys.path[0])
data_path = os.path.join(root_path, 'data')
plot_path = os.path.join(root_path, 'plot')
# pd.set_option('display.max_rows', None)

def data_preparation(machine_data : pd.DataFrame):
    # create an cersored column based on the event column
    machine_data['Censored'] = machine_data['Event'].map({'failure': 'No', 'PM':'Yes'})
    # Calculate the duration by time difference of a time compared with the time in the previous row
    # fill the nan values (from the first row) with its time value 
    machine_data['Duration'] = machine_data['Time'].diff().fillna(machine_data['Time'].iloc[0])

    # foreach 0 (due to duplicated durations) replace with nan to fill the blank cells with the duration above it
    machine_data['Duration'] = machine_data['Duration'].replace(0, np.nan).ffill()
    # sort the durations from low to high 
    machine_data = machine_data.sort_values(by='Duration', ascending=True)
    
    # sort duplicated durations such that PM events comes before the failure events
    # source: https://stackoverflow.com/questions/67845362/sort-pandas-df-subset-of-rows-within-a-group-by-specific-column 
    # source: https://sparkbyexamples.com/pandas/pandas-groupby-sort-within-groups/
    # group the data by duration and sort each group of durations from high to low and reset the index. 
    machine_data = machine_data.groupby('Duration', group_keys=True).apply(lambda x: x.sort_values(by=['Event'], ascending=False))
    machine_data = machine_data.reset_index(drop=True)

    return machine_data

# 2. Kaplan-Meier estimator for updating the probablities based on observed events 
def update_probabilities(prepared_data : pd.DataFrame):
    for index, row in prepared_data.iterrows():
        if row['Event'] == 'PM':
            # get the remaining rows underneath the current row
            remaining_rows = prepared_data.iloc[index+1:]
            # get the number of remaining rows
            num_remaining_rows = len(remaining_rows)
            # get the probability to distribute over the remaining rows
            prob_to_distribute = row['Probability']
            # spread the probability to distribute evenly over the number of remaining rows 
            prob_to_distribute_per_row = prob_to_distribute / num_remaining_rows
            remaining_rows.loc[:, 'Probability'] += prob_to_distribute_per_row
            # set the probability of the current row to zero
            prepared_data.at[index, 'Probability'] = 0
    
    return prepared_data

def create_kaplanmeier_data(prepared_data : pd.DataFrame):
    # 1. Add a column named probability to the dataframe
    prepared_data['Probability'] = 1/len(prepared_data)

    # 2. Update proabilities based on the observed events 
    prepared_data = update_probabilities(prepared_data)

    # 3. Merge duplicated, failure (censored == No) durations 
    # group data by duration and sum the probability for each group 
    grouped = prepared_data.groupby('Duration').agg({'Probability': 'sum'}).reset_index()

    # map the summed probabilities of the duration to an summed probability column
    prepared_data['summed_prob'] = prepared_data['Duration'].map(grouped.set_index('Duration')['Probability'])

    # create an filter taking all the duplicated durations that are associated with an failure, and use this filter to change the current probablity to the summed probability
    mask = (prepared_data.duplicated('Duration', keep=False)) & (prepared_data['Event'] == 'failure') #mark all duplicates
    prepared_data.loc[mask, 'Probability'] = prepared_data.loc[mask, 'summed_prob']

    # create an filter taking the duplicated durations except the first one that are associated with an failure,and use this filter to change other durations to 0 
    mask2 = (prepared_data.duplicated('Duration')) & (prepared_data['Event'] == 'failure') # mark all dups besides the first one
    prepared_data.loc[mask2, 'Probability'] = 0

    # remove the summed probability column 
    prepared_data.drop(columns='summed_prob', axis=1, inplace=True)

    # 4. Calculate the reliability function for each duration 
    # start with an reliability of 1 (100%)
    reliability = 1
    # foreach row in data, check if event is an failure, if so decrement the proability of the reliablity and set it as the new reliablity for that row
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
    plt.close()

#Weibull distribution fitting
def fit_weibull_distribution(prepared_data : pd.DataFrame):
    # 1. Create a variable with the search ranges for lambda and kappa
    l_range = np.linspace(start=1, stop=35, num=35)
    k_range = np.linspace(start=0.1, stop=3.5, num=35)

    # 2. Create a dataframe which will contain your likelihood data
    # Create pairs foreach Kappa and Lambda value
    # source: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays 
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
    # get the optimal Lambda and Kappa based on the max Loglikelihood sum
    max_loglikelihood_sum = df_weibull['Loglikelihood_sum'].max() 
    best_l_k = df_weibull[(df_weibull['Loglikelihood_sum'] == max_loglikelihood_sum)]
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
    ax.step(KM_data['Duration'], KM_data['Reliability'], label='Kaplan-Meier')
    ax.plot(weibull_data['t'], weibull_data['R_t'], label='Weibull')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join(plot_path, f'{student_nr}-Machine-{machine_name}-Reliability.png'))
    plt.close()

# Create a plot of the cost rates
def plot_cost_rates(maintenance_data : pd.DataFrame, machine_name):
    opt_idx = maintenance_data.index[maintenance_data['cost_rate'] == maintenance_data['cost_rate'].min()].values[0]
    optimal_cost_rate = maintenance_data.iloc[opt_idx]['cost_rate']
    optimal_t = maintenance_data.iloc[opt_idx]['t']

    fig, ax = plt.subplots()
    ax.set_title(f'Maintenance age impact on cost for machine {machine_name}')
    data_to_plot = maintenance_data[maintenance_data['cost_rate'] <= optimal_cost_rate+10]
    ax.plot(data_to_plot['t'], data_to_plot['cost_rate'], label='cost')
    ax.scatter(x=optimal_t, y=optimal_cost_rate, marker='.', color='gray', s=200)
    ax.vlines(x=optimal_t, ymin=0, ymax=optimal_cost_rate, color='gray', linestyles='dashed', label='T optimal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cost')
    ax.legend(['cost rate', 'T optimal'])
    plt.savefig(os.path.join(plot_path, f'{student_nr}-Machine-{machine_name}-Costs.png'))
    plt.close()

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
    cumulative_sum = maintenance_data['R_t'].cumsum()
    maintenance_data['riemann_sum'] = cumulative_sum * delta

    # 5. Calculate the cost rate for each maintenance age
    maintenance_data['cost_rate'] = maintenance_data['MCPC'] / maintenance_data['riemann_sum']

    # 6. Create a plot of cost rates
    plot_cost_rates(maintenance_data, machine_name)

    # 7. determine the optimal maintenance age and the corresponding cost rate 
    optimal = maintenance_data[maintenance_data['cost_rate'] == maintenance_data['cost_rate'].min()]
    return optimal['t'].values[0], optimal['cost_rate'].values[0]

def CBM_data_preperation(condition_data : pd.DataFrame):
    condition_data['Increments'] = condition_data['Condition'].diff()
    condition_data = condition_data[(condition_data['Increments'].notna()) & (condition_data['Increments'] > 0)].reset_index()
    return condition_data

def CBM_create_simulations(CBM_prepared_data : pd.DataFrame, failure_level, threshold):
    n_of_simulations = 10000
    simulation_data = pd.DataFrame()
    for i in range(n_of_simulations):
        state = 0
        time = 0
        while True:
            state += np.random.choice(CBM_prepared_data['Increments'])
            time += 1

            if state > failure_level:
                simulation_data.at[i, 'Duration'] = time
                simulation_data.at[i, 'Event'] = 'failure'
                break
            if state > threshold:
                simulation_data.at[i, 'Duration'] = time
                simulation_data.at[i, 'Event'] = 'PM'
                break     
    return simulation_data

def CBM_analyse_costs(sample_data : pd.DataFrame, PM_cost, CM_cost):
    # calculate the mean cycle length by taking the mean of all durations
    mean_cycle_length = sample_data['Duration'].mean()
    # Get the number of failure events
    failure_cycles = len(sample_data[sample_data['Event'] == 'failure'])
    # Get the number of Preventive maintenance events
    pm_cycles = len(sample_data[sample_data['Event'] == 'PM'])
    # Get the total number of events
    simulated_cycles = len(sample_data)
    # Calculate the mean cost per cycle based on PM cost * PM events + CM cost * CM events
    mean_cost_per_cycle = PM_cost * (pm_cycles/simulated_cycles) + CM_cost * (failure_cycles/simulated_cycles)
    # Calculate the cost rate
    cost_rate = mean_cost_per_cycle / mean_cycle_length
    return cost_rate

def CBM_create_cost_data(prepared_condition_data : pd.DataFrame, PM_cost, CM_cost, failure_level, machine_name):
    CBM_cost_data = pd.DataFrame()
    thresholds = np.arange(0, 51)
    percentiles = np.percentile(thresholds, np.arange(0, 100)).tolist()
    for threshold in thresholds:
        if threshold in percentiles:
            percentage = "{:.0%}".format(threshold / thresholds.max())
            print(f'CBM Analysis at {percentage}', end='\r', flush=True)
        sample_data = CBM_create_simulations(prepared_condition_data, failure_level, threshold)
        cost_rate = CBM_analyse_costs(sample_data, PM_cost, CM_cost)
        CBM_cost_data.at[threshold, 'cost_rate'] = cost_rate
    # get the optimal maintenance threshold & cost rate
    optimal = CBM_cost_data[CBM_cost_data['cost_rate'] == CBM_cost_data['cost_rate'].min()]

    N = 10
    fig, ax = plt.subplots()
    CBM_cost_data[N:].plot(
        title=f'Cost rate for different maintenance thresholds machine {machine_name}', 
        xlim=(N,thresholds[-1]), 
        ylim=(0, math.ceil(CBM_cost_data[N:]['cost_rate'].max())),
        ax=ax
    )   

    ax.scatter(x=optimal.index, y=optimal['cost_rate'], marker='.', color='gray', s=200)
    ax.vlines(x=optimal.index, ymin=0, ymax=optimal['cost_rate'], color='gray', linestyles='dashed')
    ax.legend(['Cost rates', 'Threshold optimal'])
    ax.set_xlabel('Maintenance threshold (M)')
    ax.set_ylabel('Cost rate')
    plt.savefig(os.path.join(plot_path, f'{student_nr}-Machine-{machine_name}-CBM-Costs.png'))
    plt.close()
    return optimal['cost_rate'].values[0], optimal.index.values[0]


def run_analysis(machine_name, PM_cost, CM_cost, analyse_CBM = 'no'):
    machine_data = pd.read_csv(os.path.join(data_path, f'{student_nr}-Machine-{machine_name}.csv'))
    prepared_data = data_preparation(machine_data)

    # Kaplan-Meier estimation
    KM_data = create_kaplanmeier_data(prepared_data)
    plot_kaplanmeier_estimation(KM_data[['Duration', 'Reliability']], machine_name)
    MTBF_KM = meantimebetweenfailures_KM(KM_data)

    # Weibull fitting
    lamb_val, kap_val = fit_weibull_distribution(prepared_data)
    weibull_data = create_weibull_curve_data(prepared_data, lamb_val, kap_val)
    MTBF_weibull = meantimebetweenfailure_weibull(lamb_val, kap_val) 
    
    print(f'Best lambda = {lamb_val} and kappa = {kap_val} values for machine {machine_name}') 
    print('The MTBF-Kaplan Meier is:', MTBF_KM)
    print('The MTBF-Weibull is:', MTBF_weibull)
    print('Cost of raw corrective maintenance is', CM_cost/MTBF_weibull)

    # Visualization
    visualization(KM_data, weibull_data, machine_name)

    # # Age-based maintenance optimization
    print('\nStart with the age-based maintenance optimization')
    best_age, best_cost_rate = create_cost_data(prepared_data, lamb_val, kap_val, PM_cost, CM_cost, machine_name)
    print('The optimal maintenance age is', best_age)
    print('The best cost rate is', best_cost_rate)

    # Condition-based maintenance
    if analyse_CBM == 'yes':
        print('\nStart with the Condition-based maintenance analysis')
        condition_data = pd.read_csv(os.path.join(data_path, f'{student_nr}-Machine-{machine_name}-condition-data.csv'))
        prepared_condition_data = CBM_data_preperation(condition_data)
        
        # Failure level
        # Failure level is the highest condition in the dataset 
        failure_level = prepared_condition_data['Condition'].max()

        CBM_cost_rate, CBM_threshold = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name) 
        print('The optimal cost rate under CBM is', CBM_cost_rate) 
        print('The optimal CBM threshold is', CBM_threshold)
    
    return

def run():
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    if not os.path.exists(data_path):
        print(f'No data folder has been found at {data_path}, \nI need data to work with first...')
        return
    while True:
        print('Hello, welcome to the Assignment 2 tool used to guide maintenance decision-making for 3 different machines')
        print('Please answer the following questions to get started:')
        try:
            analyse_CBM = 'no'
            machine_name = int(input('What machine should be analysed (1,2 or 3?): ').lower())
            if machine_name not in (1,2,3):
                print('You can only choose between machines 1, 2, or 3\n')
                continue
            PM_cost = int(input('What is the PM cost?: '))
            CM_cost = int(input('What is the CM cost?: '))
            if machine_name == 3:
                analyse_CBM = input('Do you want to include a condition-based maintenance analysis for machine 3? (yes/no) ').lower()
            print(f'Oke, let start with the analysis of machine {machine_name} using these input values')
            run_analysis(machine_name, PM_cost, CM_cost, analyse_CBM)
            print('Analysis done! See the plot folder for graphs. \n')
        except ValueError:
            print('Error within the input field, try again\n')
            continue
        except Exception as e:
            print('Crash error something bad happend during the analysis, I am closing now with error:', e)
            break

# run the program
run()








