import os
import pandas as pd
import numpy as np
import seaborn as sn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm



def test_retest_analysis(path_to_data, task_list,norm_tasks, number_of_tasks,min_effect_size, optional_task_names=None, optional_domain_names=None):
    sample_size =[]
    mean_1 = []
    mean_2 = []
    p_value = []
    r_value = []
    task_names = []
    effect_size = []
    tost_test = []
    normality_test = []


    df = pd.read_csv(path_to_data)
    df = df.loc[:,np.hstack(('ID',task_list))]

    for index, id in enumerate(task_list):
        
        if (index % 2) == 0:
            continue
        
        if df.columns[index] == 'blocks_s1':
            exclude_df = pd.read_excel('IC3_datesBlocks.xlsx')
            df_temp = pd.merge(df, exclude_df, on='ID', how='left')
            df_temp = df_temp.loc[df_temp.Exclude == 0,:]
            df_temp = df_temp.dropna(subset=[df_temp.columns[index], df_temp.columns[index+1]])
            mean1_temp = np.nanmean(df_temp.iloc[:,index])
            mean2_temp = np.nanmean(df_temp.iloc[:,index+1])
        else:    
            df_temp = df.dropna(subset=[df.columns[index], df.columns[index+1]])
            mean1_temp = np.nanmean(df_temp.iloc[:,index])
            mean2_temp = np.nanmean(df_temp.iloc[:,index+1])
        
        n_size = len(df_temp)
        
        # Effect size    
        
        std_pooled = np.sqrt( ( (n_size-1) * (np.nanstd(df_temp.iloc[:,index+1])**2) + (n_size-1) * (np.nanstd(df_temp.iloc[:,index])**2) )/((n_size * 2) - 2) )
        min_magnitude = std_pooled*min_effect_size
        cohensD = (np.nanmean(df_temp.iloc[:,index+1])- np.nanmean(df_temp.iloc[:,index]))/std_pooled
        
        norm_temp = norm_tasks[int((index-1)/2)]
        
        if norm_temp == 'No':
            
            # Wilcoxon
            _, p_temp = sp.stats.wilcoxon(df_temp.iloc[:,index],df_temp.iloc[:,index+1], correction=True) 
            
            # Spearman
            corr_temp, _ = sp.stats.spearmanr(df_temp.iloc[:,index],df_temp.iloc[:,index+1])
            
            # TOST
            _, p_greater = sp.stats.wilcoxon(df_temp.iloc[:,index] + min_magnitude, df_temp.iloc[:,index+1], alternative='greater',correction=True) 
            _, p_less = sp.stats.wilcoxon(df_temp.iloc[:,index]- min_magnitude, df_temp.iloc[:,index+1], alternative='less',correction=True)
            p_tost = max(p_greater, p_less)
            
        elif norm_temp == 'Yes':
            
            # T-test
            _, p_temp = sp.stats.ttest_rel(df_temp.iloc[:,index], df_temp.iloc[:,index+1]) 
            
            # Pearson
            corr_temp, _ = sp.stats.pearsonr(df_temp.iloc[:,index],df_temp.iloc[:,index+1])
            
            # TOST
            _, p_greater = sp.stats.ttest_rel(df_temp.iloc[:,index] + min_magnitude, df_temp.iloc[:,index+1], alternative='greater') 
            _, p_less = sp.stats.ttest_rel(df_temp.iloc[:,index]- min_magnitude, df_temp.iloc[:,index+1], alternative='less')
            p_tost = max(p_greater, p_less)
            
        
        sample_size.append(n_size)
        mean_1.append(mean1_temp)
        mean_2.append(mean2_temp)
        p_value.append(p_temp)
        r_value.append(corr_temp)
        tost_test.append(p_tost/(number_of_tasks-1))
        effect_size.append(cohensD)
        normality_test.append(norm_temp)
        task_names.append(id[:-3])
        

    p_value = np.round(sm.stats.fdrcorrection(p_value)[1].astype(np.float), 5)
    summary_table = np.transpose(np.vstack((task_names, sample_size, np.round(mean_1,2), np.round(mean_2,2), np.round(p_value,2), np.abs(np.round(effect_size,2)), np.round(tost_test,2), np.round(r_value,2), normality_test)))


    header = ['Task','N', 'Mean T1', 'Mean T2', 'P-value', 'Effect size', 'Test of equivalence', 'R-value', 'Normally Distributed']

    summary_table = np.vstack([header,summary_table])
    summary_table = pd.DataFrame(summary_table, columns=header)
    summary_table.drop(index=0, inplace=True)
    summary_table.reset_index(drop=True, inplace=True)
    summary_table['Test of equivalence'] = summary_table['Test of equivalence'].astype(str)
    summary_table = summary_table.replace({'0.0':'<0.001'})
    
    if optional_domain_names != None:
        summary_table['Domain']  = optional_domain_names
    if optional_task_names != None:
        summary_table['Task']  = optional_task_names 

    return summary_table
    