
import os
import pandas as pd
import numpy as np
import seaborn as sn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm


def environment_analysis(path_to_data, task_list,norm_tasks, number_of_tasks,min_effect_size, optional_task_names=None, optional_domain_names=None):
    
    """
    Analyze environmental differences between remote and supervised test conditions.

    This function compares performance on a set of tasks between remote and supervised conditions.
    For each paired task (assumed to be provided in task_list), the function computes the sample size,
    mean performance for remote and supervised groups, effect sizes (Cohen's d), and conducts statistical 
    tests (Wilcoxon or paired t-test based on normality) along with equivalence tests using a TOST procedure.

    Parameters
    ----------
    path_to_data : str
        Path to the CSV file containing the dataset.
    task_list : list of str
        List of task column names to analyze. These columns should be arranged in pairs (e.g., one for each session).
    norm_tasks : list of str
        List indicating whether each task pair's data is normally distributed. Each entry should be either 'Yes'
        or 'No', corresponding to each task pair.
    number_of_tasks : int
        Total number of tasks; used to adjust the TOST p-value.
    min_effect_size : float
        Minimum effect size threshold expressed as a multiplier of the pooled standard deviation for equivalence testing.
    optional_task_names : list of str, optional
        Optional list of task names to override the default task names derived from task_list.
    optional_domain_names : list of str, optional
        Optional list of domain names to include in the summary table.

    Returns
    -------
    pandas.DataFrame
        A summary table of the analysis with the following columns:
        - Domain: Domain name for each task.
        - Task: Task name.
        - N: Sample size (number of observations in the remote condition).
        - Mean Remote: Mean performance in the remote condition.
        - Mean Supervised: Mean performance in the supervised condition.
        - P-value: Corrected p-value from the statistical test.
        - Effect size: Absolute Cohen's d effect size.
        - Test of equivalence: Corrected p-value for the equivalence test (TOST).
        - Normally Distributed: Indicator of whether the data were assumed to be normally distributed.

    Examples
    --------
    >>> summary = environment_analysis("data.csv", ["task1_s1", "task1_s2"], ["Yes"], 1, 0.2)
    >>> print(summary.head())
    """
    
    # Initialize lists to store computed metrics for each task pair
    sample_size =[]
    mean_1 = []
    mean_2 = []
    p_value = []
    task_names = []
    effect_size = []
    tost_test = []
    normality_test = []
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path_to_data)
    
    # Select only the columns corresponding to ID, tasks, and remote/supervised indicators
    df = df.loc[:,np.hstack(('ID',task_list, 'Remote_s1','Remote_s2'))]

    # Loop through each task in task_list; tasks are expected to be in pairs
    for index, id in enumerate(task_list):
        
        if (index % 2) == 0:
            continue
        
        # Create a temporary DataFrame and drop rows with missing values for the current task pair
        df_temp = df.copy() 
        df_temp = df.dropna(subset=[df.columns[index], df.columns[index+1]])
        
        # Separate remote and supervised data based on indicator columns 'Remote_s1' and 'Remote_s2'
        df_remote = np.hstack((df_temp.loc[df_temp.Remote_s1 == 1,df_temp.columns[index]], df_temp.loc[df_temp.Remote_s2 == 1,df_temp.columns[index+1]]))
        df_supervised = np.hstack((df_temp.loc[df_temp.Remote_s1 == 0,df_temp.columns[index]], df_temp.loc[df_temp.Remote_s2 == 0,df_temp.columns[index+1]]))
        
        # Compute the mean for remote and supervised groups, ignoring NaN values
        mean1_temp = np.nanmean(df_remote)
        mean2_temp = np.nanmean(df_supervised)
        
        # Determine sample size based on the remote group data
        n_size = len(df_remote)
        
        # Compute Cohen's d effect size (difference divided by pooled std)
        std_pooled = np.sqrt( ( (n_size-1) * (np.nanstd(df_remote)**2) + (n_size-1) * (np.nanstd(df_supervised)**2) )/((n_size * 2) - 2) )
        min_magnitude = std_pooled*min_effect_size
        cohensD = (np.nanmean(df_remote)- np.nanmean(df_supervised))/std_pooled

        norm_temp = norm_tasks[int((index-1)/2)]
        
        if norm_temp == 'No':
            
            # Wilcoxon
            _, p_temp = sp.stats.wilcoxon(df_remote,df_supervised, correction=True) 
            
            
            # TOST
            _, p_greater = sp.stats.wilcoxon(df_remote + min_magnitude, df_supervised, alternative='greater',correction=True) 
            _, p_less = sp.stats.wilcoxon(df_remote- min_magnitude, df_supervised, alternative='less',correction=True)
            p_tost = max(p_greater, p_less)
            
        elif norm_temp == 'Yes':
            
            # T-test
            _, p_temp = sp.stats.ttest_rel(df_remote, df_supervised) 
            
            
            # TOST
            _, p_greater = sp.stats.ttest_rel(df_remote + min_magnitude, df_supervised, alternative='greater') 
            _, p_less = sp.stats.ttest_rel(df_remote - min_magnitude, df_supervised, alternative='less')
            p_tost = max(p_greater, p_less)
            
        # Append computed metrics to their respective lists
        sample_size.append(n_size)
        mean_1.append(mean1_temp)
        mean_2.append(mean2_temp)
        p_value.append(p_temp)
        tost_test.append(p_tost/(number_of_tasks-1))
        effect_size.append(cohensD)
        normality_test.append(norm_temp)
        task_names.append(id[:-3])

    # Correct p-values for multiple comparisons using FDR correction and round the values
    p_value = np.round(sm.stats.fdrcorrection(p_value)[1].astype(np.float), 5)
    
    # Combine all computed metrics into a summary table (each row corresponds to a task pair)
    summary_table = np.transpose(np.vstack((task_names,task_names, sample_size, np.round(mean_1,2), np.round(mean_2,2), np.round(p_value,2), np.abs(np.round(effect_size,2)), np.round(tost_test,2), normality_test)))

    # Define header names for the summary table
    header = ['Domain','Task','N', 'Mean Remote', 'Mean Supervised', 'P-value', 'Effect size', 'Test of equivalence', 'Normally Distributed']
    summary_table = np.vstack([header,summary_table])
    summary_table = pd.DataFrame(summary_table, columns=header)
    summary_table.drop(index=0, inplace=True)
    summary_table.reset_index(drop=True, inplace=True)

    # Convert statistical test columns to string for formatting purposes
    summary_table['Test of equivalence'] = summary_table['Test of equivalence'].astype(str)
    summary_table['P-value'] = summary_table['P-value'].astype(str)
    summary_table.replace({'Test of equivalence': {'0.0':'<0.001'}, 'P-value': {'0.0':'<0.001'}}, inplace=True)
    
    # Override the Domain and Task columns if optional names are provided
    if optional_domain_names != None:
        summary_table['Domain']  = optional_domain_names
    if optional_task_names != None:
        summary_table['Task']  = optional_task_names 
    
    # Return the final summary table DataFrame
    return summary_table

