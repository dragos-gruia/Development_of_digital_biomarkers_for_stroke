import pandas as pd
import numpy as np
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.stats.multitest import fdrcorrection


def format_and_summarise_data(path_to_clinical_data, path_to_deviation, task_list, optional_task_names=None):
    """
    Load, format, and summarise clinical and deviation data, and merge with task performance metrics.

    Parameters
    ----------
    path_to_clinical_data : str
        Path to the clinical data CSV file.
    path_to_deviation : str
        Path to the deviation data CSV file.
    task_list : list of str
        List of task performance metric column names in the merged DataFrame.
    optional_task_names : list of str, optional
        Optional list of new names for the columns specified in `task_list`. If provided, these columns
        are renamed accordingly and the task list is updated.

    Returns
    -------
    tuple of (pandas.DataFrame, list of str)
        A tuple containing:
        - df_final: The formatted and summarised DataFrame.
        - task_list: The (possibly updated) list of task names.

    Examples
    --------
    >>> df_summary, tasks = format_and_summarise_data("clinical.csv", "deviation.csv", 
    ...                                               ["srt", "trailAll"], ["ReactionTime", "TrailScore"])
    >>> df_summary.head()
    """
    
    # Load clinical data
    df_moca = pd.read_csv(path_to_clinical_data, dtype={'ID': str})
    df_moca.dropna(subset=['Moca total (30)'], inplace=True)
    
    # Select columns from 'ID' to 'user_id' and from 'age' to 'LDL cholesterol'
    df_moca = pd.concat([df_moca.loc[:, 'ID':'user_id'], df_moca.loc[:, 'age':'LDL cholesterol']], axis=1)
    df_moca.drop_duplicates(subset='ID', keep='last', inplace=True)

    # Load deviation data
    dfe = pd.read_csv(path_to_deviation)

    # Merge clinical and deviation data on 'user_id'
    df_final = df_moca.merge(dfe, on='user_id', how='left')
    df_final.reset_index(drop=True, inplace=True)
    df_final = df_final.loc[df_final['naming'].notna() | df_final['srt'].notna(), :]

    # Reverse sign of 'srt' and 'trailAll' columns
    df_final['srt'] *= -1
    df_final['trailAll'] *= -1

    # Optionally, rename task columns if alternative names are provided.
    if optional_task_names is not None:
        # Extract a copy of the columns in task_list.
        df_temp = df_final.loc[:, task_list].copy()
        df_temp.columns = optional_task_names

        # Find the first occurrence index among the columns in task_list.
        all_cols = list(df_final.columns)
        indices = {col: all_cols.index(col) for col in task_list if col in all_cols}
        if indices:
            start_index = np.min(list(indices.values()))
            # Replace columns from the start_index with the renamed task columns.
            df_final = df_final.iloc[:, :start_index].join(df_temp)
        # Update task_list to the new names.
        task_list = optional_task_names

    # Print summary statistics.
    print(f"Mean Age {np.nanmean(df_final['age']).round(2)}, and STD {np.nanstd(df_final['age']).round(2)}")
    print(f"Proportion of women is {np.sum(df_final.gender) / len(df_final)}")
    print(f"Mean MoCA {np.nanmean(df_final['Moca total (30)'])}, and STD {np.nanstd(df_final['Moca total (30)'])}")
    print(f"Proportion of patients in acute stage {np.sum(df_final['days_since_stroke'] < 14) / len(df_final)}")
    median_acute = df_final.loc[df_final['days_since_stroke'] < 14, 'days_since_stroke'].median()
    iqr_acute = st.iqr(df_final.loc[df_final['days_since_stroke'] < 14, 'days_since_stroke'])
    print(f"Median acute days since stroke {median_acute} and IQR {iqr_acute}")
    median_nonacute = df_final.loc[df_final['days_since_stroke'] >= 14, 'days_since_stroke'].median()
    iqr_nonacute = st.iqr(df_final.loc[df_final['days_since_stroke'] >= 14, 'days_since_stroke'])
    print(f"Median non-acute days since stroke {median_nonacute} and IQR {iqr_nonacute}")

    return df_final, task_list
    

def group_difference_analysis(df_final, task_list, optional_domain_names=None):
    """
    Analyze group differences for a set of tasks using non-parametric permutation tests.

    This function performs one-sample permutation test on the specified task columns in the 
    input DataFrame, testing whether the mean of each task differs significantly 
    from zero (mean of controls). Results are corrected for multiple comparisons.

    Parameters
    ----------
    df_final : pandas.DataFrame
        DataFrame containing task performance metrics.
    task_list : list of str
        List of column names corresponding to the tasks to be analyzed.
    optional_domain_names : list of str, optional
        Optional list of domain names corresponding to each task in `task_list`. If provided,
        a new column 'Domain' is added to the summary DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame summarizing the analysis with the following columns:
        - 'Mean': Mean value of the task metric.
        - 'STD': Standard deviation of the task metric.
        - 'P-value': FDR-corrected p-value from the one-sample t-test (testing if the mean 
          is different from zero). P-values of 0.000 are replaced with "<0.001".
        - 'Effect size': Absolute Cohen's d (mean divided by standard deviation).
        - 'Task': Name of the task.
        - 'Domain': (Optional) Domain name for the task if `optional_domain_names` is provided.

    Examples
    --------
    >>> df_summary = group_difference_analysis(df, ["srt", "trailAll"])
    >>> print(df_summary.head())
          Mean    STD   P-value  Effect size       Task
    0    -0.35   0.45     0.023          0.78       srt
    1     0.12   0.38     0.156          0.32  trailAll
    """
    
    # Subset the DataFrame to include only the task columns.
    df_tasks = df_final.loc[:, task_list]

    # Calculate the mean and standard deviation for each task.
    means = df_tasks.mean(skipna=True)
    stds = df_tasks.std(skipna=True)

    # Calculate Cohen's d as |mean / std|
    effect_sizes = (means / stds).abs().round(2)

    # Compute one-sample permutation test for each task column.
    pvals = df_tasks.apply(lambda col: permutation_test_mean(col.dropna()))
    
    # Apply FDR correction to the p-values.
    _, pvals_corrected = fdrcorrection(pvals, alpha=0.05)
    pvals_corrected = np.round(pvals_corrected, 3)
    pvals_series = pd.Series(pvals_corrected, index=df_tasks.columns).replace(0.000, '<0.001')

    # Create a summary DataFrame.
    df_results = pd.DataFrame({
        'Mean': means.round(2),
        'STD': stds.round(2),
        'P-value': pvals_series,
        'Effect size': effect_sizes
    })
    df_results['Task'] = df_results.index
    df_results.reset_index(drop=True, inplace=True)

    if optional_domain_names is not None:
        df_results['Domain'] = optional_domain_names

    return df_results

def permutation_test_mean(data,mu0=0,n_permutations=50_000,alternative='two-sided',random_state=42):
    """
    One-sample permutation test for the mean (sign-flip method).

    Parameters
    ----------
    data : array-like
        Your sample of observations.
    mu0 : float, default 0
        Null-hypothesis mean to test against.
    n_permutations : int, default 50000
        How many random sign flips to generate.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Tail of the test.
    random_state : int or None
        RNG seed for reproducibility.

    Returns
    -------
    float
        p-value of the permutation test.
    """
    
    rng = np.random.default_rng(random_state)
    x = np.asarray(data) - mu0          # shift so H0 is "mean == 0"
    obs = x.mean()                      # observed test statistic

    # Generate a matrix of random ±1 with shape (n_permutations, n_obs)
    signs = rng.choice((-1, 1), size=(n_permutations, x.size))
    perm_means = (signs * x).mean(axis=1)

    if alternative == 'two-sided':
        p = (np.abs(perm_means) >= abs(obs)).mean()
    elif alternative == 'greater':      # H1: mean > mu0
        p = (perm_means >= obs).mean()
    else:                               # 'less', H1: mean < mu0
        p = (perm_means <= obs).mean()

    # add the observed case itself for an unbiased estimate
    p = (p * n_permutations + 1) / (n_permutations + 1)
    return p


def plot_group_difference_analysis(df_final, task_list, optional_domain_names):
    
    """
    Plot a barplot showing group differences for a set of tasks.

    Parameters
    ----------
    df_final : pandas.DataFrame
        DataFrame containing at least a 'user_id' column and columns corresponding to the tasks
        specified in task_list.
    task_list : list of str
        List of task column names whose scores are to be plotted.
    optional_domain_names : list or array-like
        List of domain names to be added to the melted DataFrame for labeling purposes.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'user_id': [1, 2, 3],
    ...     'task1': [0.5, -0.2, 0.1],
    ...     'task2': [0.3, -0.1, 0.0]
    ... })
    >>> task_list = ['task1', 'task2']
    >>> optional_domain_names = ['Domain1'] * 3
    >>> plot_group_difference_analysis(df, task_list, optional_domain_names)
    """
    
    # Keep only the 'user_id' and the specified task columns from the DataFrame.
    df_final = pd.concat([df_final.loc[:,'user_id'], df_final.loc[:,task_list]], axis=1)
    
    # Create a DataFrame with domain labels by repeating the optional domain names for each row.
    domain_long = pd.DataFrame(np.repeat(optional_domain_names, len(df_final)),columns=['Domain'])
    
    # Reshape the DataFrame from wide to long format for easier plotting with seaborn.
    df_final = pd.melt(df_final,id_vars='user_id', value_vars=task_list, var_name='Task', value_name='Score')
    df_final = pd.concat([df_final, domain_long], axis=1)

    # Plot a barplot of the scores for each task with error bars indicating the confidence interval.
    fig_dims=(45,15)
    sn.set_context("talk",font_scale=2.5)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    sn.barplot(x='Task',y='Score',  palette=['#7851A9'], dodge=False,
                data=df_final, ax=ax,errorbar='ci', capsize=.4)
    sn.set_style(style='ticks') 
    _ = plt.xticks(rotation=90)
    ax.set(xlabel=None) 
    ax.set_ylim(-10, 2)
    ax.set_ylabel("Deviation from expected in SD units", fontsize=35)
    ax.axhline(y=0,color='black',linestyle='-',label='Controls mean')

    # Add text annotations and vertical lines to demarcate different cognitive domains.
    plt.text(3.65,0.5,'Memory',color='black',fontsize=22)
    ax.axvline(x=4.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(8.5,0.5,'Language',color='black',fontsize=22)
    ax.axvline(x=9.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(12.5,0.5,'Executive',color='black',fontsize=22)
    ax.axvline(x=13.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(15.6,0.5,'Attention',color='black',fontsize=22)
    ax.axvline(x=16.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(17.85,0.5,'Motor',color='black',fontsize=22)
    ax.axvline(x=18.5,color='black',linestyle=(0, (5, 5)))

    plt.text(18.55,0.5,'Arithmetics',color='black',fontsize=22)
    ax.axvline(x=19.5,color='black',linestyle=(0, (5, 5)))

    plt.text(19.85,0.5,'Praxis',color='black',fontsize=22)

    # Adjust the width of each bar in the barplot and recenter them.
    for p in ax.patches:
        new_value = 0.8
        current_width = p.get_width()
        diff = current_width - new_value
        p.set_width(new_value)
        p.set_x(p.get_x() + diff * .5)
        
    plt.savefig('plots/comparison_between_healthy_and_patients_dfe.png', format='png', transparent=True, bbox_inches='tight')


def plot_sensitivity_analysis(df_final, task_list,  impairment_type='any impairment'):
    
    """
    Plot sensitivity analysis for patient impairment in tasks.

    This function computes the percentage of patients who are impaired on each task,
    based on predefined thresholds (e.g., 'any impairment', 'mild', 'moderate', 'severe').

    Parameters
    ----------
    df_final : pandas.DataFrame
        DataFrame containing patient data. It must include columns for task scores, 
        'Moca total (30)', and 'days_since_stroke'.
    task_list : list of str
        List of task column names for which the sensitivity analysis is to be performed.
    impairment_type : str, optional
        The type of impairment threshold to plot. Options include 'any impairment', 'mild', 
        'moderate', or 'severe'. Default is 'any impairment'.

    Returns
    -------
    None
        This function creates a barplot and does not return any value.

    Examples
    --------
    >>> plot_sensitivity_analysis(df, ['task1', 'task2'], impairment_type='mild')
    """
    
    # Filter data 
    df_simplified = df_final[df_final['Moca total (30)']>=26]
    df_chronic = df_simplified[df_simplified.days_since_stroke >14]
    
    # Initialize lists to store impairment percentages for the overall patient group
    mild = []
    moderate = []
    severe = []
    all_impairments = []
    
    # Loop through each task in the task list to compute impairment percentages
    for task in task_list:
        
        all_impairments.append(np.sum(df_simplified[task]<=-1.5)/len(df_simplified))
        mild.append(np.sum( (df_simplified[task]<=-1.5) & (df_simplified[task]>-2) )/len(df_simplified) )
        moderate.append(np.sum((df_simplified[task]<=-2) & (df_simplified[task]>-2.5) ) /len(df_simplified)  )
        severe.append(np.sum(df_simplified[task]<=-2.5) /len(df_simplified)  )
        
    # Create a summary DataFrame from the computed percentages (rounded to 2 decimals)
    summaries = pd.DataFrame(np.transpose([task_list, np.round(all_impairments, 2), 
                                             np.round(mild, 2), np.round(moderate, 2), 
                                             np.round(severe, 2)]))
    summaries.columns = ['task','any impairment', 'mild','moderate','severe']
 
    # Reshape the summary DataFrame to long format for plotting
    summaries_temp = summaries.melt(id_vars=['task'], 
                                    value_vars=['any impairment', 'severe', 'moderate', 'mild'])
    
    # Convert values to float and scale them to percentages (0-100)
    summaries_temp.value = summaries_temp.value.astype(float)
    summaries_temp.value = summaries_temp.value.multiply(100).astype(int)
    # Filter the DataFrame for the specified impairment type
    summaries_temp = summaries_temp[summaries_temp.variable == impairment_type]
    
    # Reinitialize lists
    mild = []
    moderate = []
    severe = []
    all_impairments = []
    
    # Loop through each task to compute impairment percentages for chronic patients
    for task in task_list:
        
        all_impairments.append(np.sum(df_chronic[task]<=-1.5)/len(df_simplified))
        mild.append(np.sum( (df_chronic[task]<=-1.5) & (df_chronic[task]>-2) )/len(df_simplified) )
        moderate.append(np.sum((df_chronic[task]<=-2) & (df_chronic[task]>-2.5) ) /len(df_simplified)  )
        severe.append(np.sum(df_chronic[task]<=-2.5) /len(df_simplified)  )
    
    # Create a summary DataFrame for chronic patients   
    summaries_chronic = pd.DataFrame(np.transpose([task_list, np.round(all_impairments, 2), 
                                                     np.round(mild, 2), np.round(moderate, 2), 
                                                     np.round(severe, 2)]))
    summaries_chronic.columns = ['task','any impairment', 'mild','moderate','severe']
    
    # Reshape the chronic summary DataFrame to long format for plotting
    summaries_chronic = summaries_chronic.melt(id_vars=['task'], value_vars=['any impairment','severe','moderate', 'mild'])
    summaries_chronic.value = summaries_chronic.value.astype(float)
    summaries_chronic.value = summaries_chronic.value.multiply(other=100).astype(int)
    summaries_chronic= summaries_chronic[summaries_chronic.variable == impairment_type] 
    
    # Set dimensions and context for the plot
    fig_dims=(44,25)
    sn.set_context("talk",font_scale=2.5)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    # Plot the overall impairment percentages as a barplot
    bar1 =sn.barplot(x='task',y='value',
                data=summaries_temp,palette=['#7851A9'], ax=ax, width=0.8)

    # Set plot style and rotate x-axis labels for readability
    sn.set_style(style='ticks') 
    _ = plt.xticks(rotation=90)
    ax.set(xlabel=None) 
    ax.set_ylim(0, 65)

    # Add text annotations and vertical lines to delineate cognitive domains
    plt.text(3.75,60,'Memory',color='black',fontsize=22)
    ax.axvline(x=4.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(8.6,60,'Language',color='black',fontsize=22)
    ax.axvline(x=9.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(12.6,60,'Executive',color='black',fontsize=22)
    ax.axvline(x=13.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(15.7,60,'Attention',color='black',fontsize=22)
    ax.axvline(x=16.5,color='black',linestyle=(0, (5, 5)),label='Controls mean')

    plt.text(17.95,60,'Motor',color='black',fontsize=22)
    ax.axvline(x=18.5,color='black',linestyle=(0, (5, 5)))

    plt.text(18.55,60,'Calculation',color='black',fontsize=22)
    ax.axvline(x=19.5,color='black',linestyle=(0, (5, 5)))

    plt.text(19.9,60,'Praxis',color='black',fontsize=22)

    # label each bar in barplot
    for p in ax.patches:
    # get the height of each bar
        height = p.get_height()
        # adding text to each bar
        ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar      
        y = height+0.65 if height>0 else height+0.65, # y-coordinate position of data label, padded 100 above bar
        s = "{:.0f}%".format(height) if height>0 else "{:.0f}%".format(height), # data label, formatted to ignore decimals
        ha = "center",
        fontsize= 20) # sets horizontal alignment (ha) to center
        
    # Plot the chronic impairment percentages as an overlay barplot with a different color palette
    bar2= sn.barplot(x='task',y='value', hue='variable', 
                data=summaries_chronic, palette=['#b098cd'], ax=ax, errorbar=None, width=0.8)
    ax.legend_.remove()
    ax.set_ylabel("Percentage of patients impaired in each task", fontsize=35)
    ax.set_xlabel("", fontsize=20)
    fig.savefig('plots/dev_from_expected_breakdown.png', format='png', transparent=True, bbox_inches='tight')


def moca_vs_ic3_analysis(df_final):
    """
    Compare MoCA and IC3 performance by computing domain sensitivities and plotting the results.

    Parameters
    ----------
    df_final : pandas.DataFrame
        DataFrame containing the raw MoCA test scores and IC3 task scores. Expected to include columns for
        individual MoCA tests such as 'Moca total (30)', 'Trail Making (1)', 'visuospatial (4)', 'Naming (3)',
        'Auditory Sustained Attention (1)', 'Arithmetic/Serial 7 (3)', 'Orientation (6)', 'Repetition (2)',
        'Fluency (1)', 'Attention- digit span (2)', 'Abstraction (2)', and 'Memory (5)', as well as IC3 tasks
        such as 'Pear Cancellation', 'Trail-making', 'Rule Learning', 'Odd One Out', 'Blocks',
        'Semantic Judgement', 'Gesture Recognition', 'Language Comprehension', 'Paired Associates Learning',
        'Task Recall', 'Spatial Span', 'Digit Span', 'Orientation', 'Choice Reaction Time',
        'Simple Reaction Time', 'Auditory Attention', and 'Graded Calculation'.

    Returns
    -------
    None
        The function produces a plot comparing domain sensitivities between MoCA and IC3 scores and does not
        return a value.

    Examples
    --------
    >>> moca_vs_ic3_analysis(df_final)
    """
    
    # Exclude participants with missing MoCA or IC3 scores.
    exclude_ids = [
        'ic3study00062-session3-versionA', 'ic3study00020-session1-versionA', 'ic3study00052-session1-versionA',
        'ic3study00076-session1-versionA', 'ic3study00077-session1-versionA', 'ic3study00078-session1-versionA',
        'ic3study00079-session1-versionA', 'ic3study00083-session1-versionA', 'ic3study00108-session1-versionA'
    ]
    df_final = df_final[~df_final.user_id.isin(exclude_ids)].copy()

    # Convert MoCA test scores to numeric values.
    moca_tests = ['Moca total (30)', 'Trail Making (1)', 'visuospatial (4)', 'Naming (3)',
                  'Auditory Sustained Attention (1)', 'Arithmetic/Serial 7 (3)', 'Orientation (6)',
                  'Repetition (2)', 'Fluency (1)', 'Attention- digit span (2)', 'Abstraction (2)', 'Memory (5)']
    df_final[moca_tests] = df_final[moca_tests].apply(pd.to_numeric, errors='coerce')

    # Replace value 5 with 4 in 'visuospatial (4)'.
    df_final['visuospatial (4)'] = df_final['visuospatial (4)'].replace(5, 4)

    # Compute MoCA domain scores.
    df_final['Visuospatial/Executive_moca'] = df_final['visuospatial (4)'] + df_final['Trail Making (1)']
    df_final['Language_moca'] = (df_final['Naming (3)'] + df_final['Repetition (2)'] +
                                 df_final['Fluency (1)'] + df_final['Abstraction (2)'])
    df_final['Memory_moca'] = df_final['Memory (5)']
    df_final['Attention_moca'] = df_final['Auditory Sustained Attention (1)'] + df_final['Attention- digit span (2)']
    df_final['Orientation_moca'] = df_final['Orientation (6)']
    df_final['Calculation_moca'] = df_final['Arithmetic/Serial 7 (3)']
    df_final['auditoryAttention_moca'] = df_final['Auditory Sustained Attention (1)']
    df_final['trail_moca'] = df_final['Trail Making (1)']

    # Convert MoCA domain scores to binary indicators.
    df_final['Visuospatial/Executive_moca'] = (df_final['Visuospatial/Executive_moca'] == 
                                               df_final['Visuospatial/Executive_moca'].max()).astype(int)
    df_final['Language_moca'] = (df_final['Language_moca'] == 
                                 df_final['Language_moca'].max()).astype(int)
    df_final['Memory_moca'] = (df_final['Memory_moca'] >= (df_final['Memory_moca'].max() - 1)).astype(int)
    df_final['Attention_moca'] = (df_final['Attention_moca'] == 
                                  df_final['Attention_moca'].max()).astype(int)
    df_final['Orientation_moca'] = (df_final['Orientation_moca'] == 
                                    df_final['Orientation_moca'].max()).astype(int)
    df_final['Calculation_moca'] = (df_final['Calculation_moca'] == 
                                    df_final['Calculation_moca'].max()).astype(int)
    df_final['auditoryAttention_moca'] = (df_final['auditoryAttention_moca'] == 
                                          df_final['auditoryAttention_moca'].max()).astype(int)
    df_final['trail_moca'] = (df_final['trail_moca'] == 
                              df_final['trail_moca'].max()).astype(int)

    # Replace NaN values in language-related subtests with 0.
    for col in ['Naming', 'Reading', 'Repetition']:
        df_final[col].fillna(0, inplace=True)
    swt_language = (df_final['Naming'] > -1.5) & (df_final['Repetition'] > -1.5) & (df_final['Reading'] > -1.5)

    # Compute binary IC3 domain indicators.
    df_final['Visuospatial/Executive_ic3'] = ((df_final['Pear Cancellation'] > -1.5) &
                                              (df_final['Trail-making'] > -1.5) &
                                              (df_final['Rule Learning'] > -1.5) &
                                              (df_final['Odd One Out'] > -1.5) &
                                              (df_final['Blocks'] > -1.5)).astype(int)
    df_final['Language_ic3'] = ((df_final['Semantic Judgement'] > -1.5) &
                                (df_final['Gesture Recognition'] > -1.5) &
                                (df_final['Language Comprehension'] > -1.5) &
                                swt_language).astype(int)
    df_final['Memory_ic3'] = ((df_final['Paired Associates Learning'] > -1.5) &
                              (df_final['Task Recall'] > -1.5) &
                              (df_final['Spatial Span'] > -1.5) &
                              (df_final['Digit Span'] > -1.5) &
                              (df_final['Orientation'] > -1.5)).astype(int)
    df_final['Attention_ic3'] = ((df_final['Choice Reaction Time'] > -1.5) &
                                 (df_final['Simple Reaction Time'] > -1.5) &
                                 (df_final['Auditory Attention'] > -1.5)).astype(int)
    df_final['Orientation_ic3'] = (df_final['Orientation'] > -1.5).astype(int)
    df_final['Calculation_ic3'] = (df_final['Graded Calculation'] > -1.5).astype(int)
    df_final['auditoryAttention_ic3'] = (df_final['Auditory Attention'] > -1.5).astype(int)
    df_final['trail_ic3'] = (df_final['Trail-making'] > -1.5).astype(int)

    # Drop rows with missing MoCA total score and duplicate IDs.
    df_final.dropna(subset=['Moca total (30)'], inplace=True)
    df_final.drop_duplicates(subset=['ID'], inplace=True)

    # Helper function for safe division.
    def safe_div(numer, denom):
        return numer / denom if denom != 0 else np.nan

    # Define domain pairs: (MoCA column, IC3 column).
    domain_pairs = [
        ('Visuospatial/Executive_moca', 'Visuospatial/Executive_ic3'),
        ('Language_moca', 'Language_ic3'),
        ('Memory_moca', 'Memory_ic3'),
        ('Attention_moca', 'Attention_ic3'),
        ('Orientation_moca', 'Orientation_ic3'),
        ('Calculation_moca', 'Calculation_ic3'),
        ('auditoryAttention_moca', 'auditoryAttention_ic3'),
        ('trail_moca', 'trail_ic3')
    ]

    # Calculate domain sensitivity based on MoCA.
    domain_sensitivity = []
    for moca_col, ic3_col in domain_pairs:
        denom = len(df_final[df_final[moca_col] == 0])
        num = sum((df_final[moca_col] == 0) & (df_final[ic3_col] == 0))
        domain_sensitivity.append(safe_div(num, denom))
    
    # Calculate domain sensitivity based on IC3 (gold standard).
    domain_sensitivity_ic3gold = []
    for moca_col, ic3_col in domain_pairs:
        denom = len(df_final[df_final[ic3_col] == 0])
        num = sum((df_final[moca_col] == 0) & (df_final[ic3_col] == 0))
        domain_sensitivity_ic3gold.append(safe_div(num, denom))
    
    # Convert sensitivities to percentages.
    domain_sensitivity = [round(x, 4) * 100 for x in domain_sensitivity]
    domain_sensitivity_ic3gold = [round(x, 4) * 100 for x in domain_sensitivity_ic3gold]

    # Create a comparison DataFrame.
    domains = ['Visuospatial/Executive', 'Language', 'Memory', 'Attention', 
               'Orientation', 'Calculation', 'Auditory Attention', 'Trail-making']
    moca_comparison = pd.DataFrame({
        'Domain': domains,
        'Sensitivity': domain_sensitivity,
        'Sensitivity_IC3gold': domain_sensitivity_ic3gold
    })

    # Reshape the comparison DataFrame to long format.
    moca_comparison_long = pd.melt(moca_comparison, id_vars='Domain',
                                   value_vars=['Sensitivity', 'Sensitivity_IC3gold'])
    moca_comparison_long.replace({'variable': {'Sensitivity': 'Sensitivity of IC3',
                                                'Sensitivity_IC3gold': 'Sensitivity of MOCA'}},
                                 inplace=True)
    moca_comparison_long.rename(columns={'variable': 'Legend'}, inplace=True)

    # Plot the comparison.
    plot_moca_vs_ic3(moca_comparison_long)


def plot_moca_vs_ic3(moca_comparison_long):
    """
    Plot a bar chart comparing MoCA and IC3 domain sensitivities.

    Parameters
    ----------
    moca_comparison_long : pandas.DataFrame
        A long-format DataFrame containing the following columns:
            - 'Domain': The name of the cognitive domain.
            - 'value': The sensitivity percentage for that domain.
            - 'Legend': A label indicating the source, either 'Sensitivity of IC3' or 
              'Sensitivity of MOCA'.

    Examples
    --------
    >>> import pandas as pd
    >>> df_long = pd.DataFrame({
    ...     'Domain': ['Memory', 'Memory', 'Attention', 'Attention'],
    ...     'value': [80, 90, 75, 85],
    ...     'Legend': ['Sensitivity of IC3', 'Sensitivity of MOCA',
    ...                'Sensitivity of IC3', 'Sensitivity of MOCA']
    ... })
    >>> plot_moca_vs_ic3(df_long)
    """
    
    # Set plot context and style.
    sn.set_context("talk", font_scale=1.5)
    sn.set_style("ticks")
    fig_dims = (30, 20)
    
    # Create figure and axis.
    fig, ax = plt.subplots(figsize=fig_dims)
    
    # Create barplot.
    sn.barplot(
        x='Domain', y='value', hue='Legend',
        palette=['#7851A9', '#fff5cc'],
        data=moca_comparison_long, ax=ax
    )
    
    # Add custom text annotations.
    plt.text(1.95, 105, 'Domain-level Comparison', color='black', fontsize=26)
    plt.text(3.60, 105, 'Task-level Comparison', color='black', fontsize=26)
    ax.axvline(x=3.5, color='black', linestyle=(0, (5, 5)))
    ax.set_ylim(0, 115)
    
    # Annotate each bar with its value.
    for patch in ax.patches:
        height = patch.get_height()
        x_pos = patch.get_x() + patch.get_width() / 2
        y_pos = height + 0.55 if height > 0 else height - 0.4
        label = f"{height:.0f}%" if height > 0 else f"{height:.0f}\n*"
        ax.text(x=x_pos, y=y_pos, s=label, ha="center", fontsize=20)
    
    # Set axis labels.
    ax.set(xlabel=None, ylabel="Percentage of impairments detected")
    
    # Save the plot.
    plt.savefig('plots/sensitivity_IC3_and_moca.png', format='png', transparent=True, bbox_inches='tight')