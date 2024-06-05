
import os
import pandas as pd
import numpy as np
import seaborn as sn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm


def environment_analysis(path_to_data, task_list,norm_tasks, number_of_tasks,min_effect_size, optional_task_names=None, optional_domain_names=None):

    sample_size =[]
    mean_1 = []
    mean_2 = []
    p_value = []
    task_names = []
    effect_size = []
    tost_test = []
    normality_test = []

    df = pd.read_csv(path_to_data)
    df = df.loc[:,np.hstack(('ID',task_list, 'Remote_s1','Remote_s2'))]

    for index, id in enumerate(task_list):
        
        if (index % 2) == 0:
            continue
        
        df_temp = df.copy() 
        df_temp = df.dropna(subset=[df.columns[index], df.columns[index+1]])
            
        df_remote = np.hstack((df_temp.loc[df_temp.Remote_s1 == 1,df_temp.columns[index]], df_temp.loc[df_temp.Remote_s2 == 1,df_temp.columns[index+1]]))
        df_supervised = np.hstack((df_temp.loc[df_temp.Remote_s1 == 0,df_temp.columns[index]], df_temp.loc[df_temp.Remote_s2 == 0,df_temp.columns[index+1]]))
        
        mean1_temp = np.nanmean(df_remote)
        mean2_temp = np.nanmean(df_supervised)
        
        n_size = len(df_remote)
        
        # Effect size    
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
            
        
        sample_size.append(n_size)
        mean_1.append(mean1_temp)
        mean_2.append(mean2_temp)
        p_value.append(p_temp)
        tost_test.append(p_tost/(number_of_tasks-1))
        effect_size.append(cohensD)
        normality_test.append(norm_temp)
        task_names.append(id[:-3])


    p_value = np.round(sm.stats.fdrcorrection(p_value)[1].astype(np.float), 5)
    summary_table = np.transpose(np.vstack((task_names,task_names, sample_size, np.round(mean_1,2), np.round(mean_2,2), np.round(p_value,2), np.abs(np.round(effect_size,2)), np.round(tost_test,2), normality_test)))

    header = ['Domain','Task','N', 'Mean Remote', 'Mean Supervised', 'P-value', 'Effect size', 'Test of equivalence', 'Normally Distributed']
    summary_table = np.vstack([header,summary_table])
    summary_table = pd.DataFrame(summary_table, columns=header)
    summary_table.drop(index=0, inplace=True)
    summary_table.reset_index(drop=True, inplace=True)

    summary_table['Test of equivalence'] = summary_table['Test of equivalence'].astype(str)
    summary_table['P-value'] = summary_table['P-value'].astype(str)
    summary_table.replace({'Test of equivalence': {'0.0':'<0.001'}, 'P-value': {'0.0':'<0.001'}}, inplace=True)

    if optional_domain_names != None:
        summary_table['Domain']  = optional_domain_names
    if optional_task_names != None:
        summary_table['Task']  = optional_task_names 
        
    return summary_table

