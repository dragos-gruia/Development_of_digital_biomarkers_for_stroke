import os
import pandas as pd
import numpy as np
import seaborn as sn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm



def format_and_summarise_data(path_to_clinical_data, path_to_deviation, task_list, optional_task_names=None):
    
    
    df_moca = pd.read_csv(path_to_clinical_data,dtype={'ID':str})
    df_moca.dropna(subset=['Moca total (30)'], inplace=True)
    df_moca = pd.concat([df_moca.loc[:,'ID':'user_id'], df_moca.loc[:,'age':'LDL cholesterol']],axis=1)
    df_moca.drop_duplicates(subset='ID', keep='last', inplace=True)
    
    dfe = pd.read_csv(path_to_deviation)
    
    df_final = df_moca.merge(dfe, on='user_id',how='left')
    df_final.reset_index(drop=True,inplace=True)
    df_final = df_final.loc[df_final['naming'].notna() | df_final['srt'].notna(),:]
    df_final.srt = df_final.srt *-1
    df_final.trailAll = df_final.trailAll *-1
    
    if optional_task_names != None:
        df_temp = df_final.loc[:,task_list].copy()
        df_temp.columns = optional_task_names
        indices = {string: (list(df_final.columns).index(string) if string in list(df_final.columns) else None) for string in task_list}
        start_index = np.min(list(indices.values()))
        df_final = df_final.iloc[:,0:start_index].join(df_temp)
        
        task_list = optional_task_names
        
        
    print(f"Mean Age {np.nanmean(df_final['age']).round(2)}, and STD {np.nanstd(df_final['age']).round(2)}")
    print(f'Proportion of women is {np.sum(df_final.gender)/len(df_final)}')
    print(f"Mean MoCA {np.nanmean(df_final['Moca total (30)'])}, and STD {np.nanstd(df_final['Moca total (30)'])}")
    print(f"Proportion of patients in acute stage {sum(df_final['days_since_stroke'] <14)/len(df_final)}")
    print(f"Median acute days since stroke {df_final.loc[df_final['days_since_stroke'] <14,'days_since_stroke'].median()} and IQR {sp.stats.iqr(df_final.loc[df_final['days_since_stroke'] <14,'days_since_stroke'])}")
    print(f"Median acute days since stroke {df_final.loc[df_final['days_since_stroke'] >=14,'days_since_stroke'].median()} and IQR {sp.stats.iqr(df_final.loc[df_final['days_since_stroke'] >=14,'days_since_stroke'])}")
    
    return df_final,task_list
    
    
def group_difference_analysis(df_final, task_list, optional_domain_names=None):
    
    df_final = df_final.loc[:,task_list] 
     
    ttest_stroke =[]
    cohen =[]
    for task in df_final.columns:
        df_results = df_final[task].dropna()
        _, p_value = sp.stats.ttest_1samp(a=df_results, popmean=0)
        std_pooled = np.sqrt( ( (len(df_results)-1) * (np.nanstd(df_results)**2) + (len(df_results)-1) * (np.nanstd(df_results)**2) )/((len(df_results) * 2) - 2) )
        cohensD = (np.nanmean(df_results)- 0)/std_pooled
        cohen.append(np.abs(np.round(cohensD,2)))
        ttest_stroke.append(p_value)
        
    ttest_stroke = np.round(sm.stats.fdrcorrection(ttest_stroke)[1],3)
    ttest_stroke = pd.Series(ttest_stroke, index=df_final.columns).replace(0.000,'<0.001')
    cohen = pd.Series(cohen, index=df_final.columns)
    df_results = pd.concat([np.round(np.mean(df_final,axis=0),2), np.round(np.std(df_final,axis=0),2), ttest_stroke, cohen],axis=1)
    df_results.columns=['Mean','STD', 'P-value','Effect size']
    df_results['Task'] = df_results.index
    df_results.reset_index(drop=True,inplace=True)
    
    if optional_domain_names != None:
        df_results['Domain'] = optional_domain_names
    
    return df_results

def plot_group_difference_analysis(df_final, task_list, optional_domain_names):
    
    
    df_final = pd.concat([df_final.loc[:,'user_id'], df_final.loc[:,task_list]], axis=1)
    
    domain_long = pd.DataFrame(np.repeat(optional_domain_names, len(df_final)),columns=['Domain'])
    df_final = pd.melt(df_final,id_vars='user_id', value_vars=task_list, var_name='Task', value_name='Score')
    df_final = pd.concat([df_final, domain_long], axis=1)

    fig_dims=(45,15)
    sn.set_context("talk",font_scale=2.5)
    
    fig, ax = plt.subplots(figsize=fig_dims)
    
    sn.barplot(x='Task',y='Score',  palette=['#7851A9'], dodge=False,
                data=df_final, ax=ax,errorbar='ci', capsize=.4)
    
    sn.set_style(style='ticks') 
    _ = plt.xticks(rotation=90)
    ax.set(xlabel=None) # X label
    ax.set_ylim(-10, 2)
    ax.set_ylabel("Deviation from expected in SD units", fontsize=35)
    ax.axhline(y=0,color='black',linestyle='-',label='Controls mean')

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

    # label each bar in barplot
    for p in ax.patches:
        
        new_value = 0.8
        current_width = p.get_width()
        diff = current_width - new_value

        # we change the bar width
        p.set_width(new_value)

        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)
        
    #plt.savefig('plots/comparison_between_healthy_and_patients_dfe.png', format='png', transparent=True, bbox_inches='tight')
    return None

def plot_sensitivity_analysis(df_final, task_list,  impairment_type='any impairment'):
    
    df_simplified = df_final[df_final['Moca total (30)']>=26]
    df_chronic = df_simplified[df_simplified.days_since_stroke >14]
    
    mild = []
    moderate = []
    severe = []
    all_impairments = []
    for task in task_list:
        
        all_impairments.append(np.sum(df_simplified[task]<=-1.5)/len(df_simplified))
        mild.append(np.sum( (df_simplified[task]<=-1.5) & (df_simplified[task]>-2) )/len(df_simplified) )
        moderate.append(np.sum((df_simplified[task]<=-2) & (df_simplified[task]>-2.5) ) /len(df_simplified)  )
        severe.append(np.sum(df_simplified[task]<=-2.5) /len(df_simplified)  )
        
    summaries = pd.DataFrame(np.transpose([task_list, np.round(all_impairments,2), np.round(mild,2), np.round(moderate,2), np.round(severe,2)]))
    summaries.columns = ['task','any impairment', 'mild','moderate','severe']
 
    summaries_temp = summaries.melt(id_vars=['task'], value_vars=['any impairment','severe','moderate', 'mild'])

    summaries_temp.value = summaries_temp.value.astype(float)
    summaries_temp.value = summaries_temp.value.multiply(other=100).astype(int)
    summaries_temp= summaries_temp[summaries_temp.variable == impairment_type]
    
    mild = []
    moderate = []
    severe = []
    all_impairments = []
    for task in task_list:
        
        all_impairments.append(np.sum(df_chronic[task]<=-1.5)/len(df_simplified))
        mild.append(np.sum( (df_chronic[task]<=-1.5) & (df_chronic[task]>-2) )/len(df_simplified) )
        moderate.append(np.sum((df_chronic[task]<=-2) & (df_chronic[task]>-2.5) ) /len(df_simplified)  )
        severe.append(np.sum(df_chronic[task]<=-2.5) /len(df_simplified)  )
        
    summaries_chronic = pd.DataFrame(np.transpose([task_list, np.round(all_impairments,2), np.round(mild,2), np.round(moderate,2), np.round(severe,2)]))
    summaries_chronic.columns = ['task','any impairment', 'mild','moderate','severe']
        
    summaries_chronic = summaries_chronic.melt(id_vars=['task'], value_vars=['any impairment','severe','moderate', 'mild'])
    summaries_chronic.value = summaries_chronic.value.astype(float)
    summaries_chronic.value = summaries_chronic.value.multiply(other=100).astype(int)
    summaries_chronic= summaries_chronic[summaries_chronic.variable == impairment_type] 
    
    fig_dims=(44,25)
    sn.set_context("talk",font_scale=2.5)
    
    fig, ax = plt.subplots(figsize=fig_dims)
    
    bar1 =sn.barplot(x='task',y='value',
                data=summaries_temp,palette=['#7851A9'], ax=ax, width=0.8)

    sn.set_style(style='ticks') 
    _ = plt.xticks(rotation=90)
    ax.set(xlabel=None) 
    ax.set_ylim(0, 65)

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

    bar2= sn.barplot(x='task',y='value', hue='variable', 
                data=summaries_chronic, palette=['#b098cd'], ax=ax, errorbar=None, width=0.8)
    ax.legend_.remove()
    ax.set_ylabel("Percentage of patients impaired in each task", fontsize=35)
    ax.set_xlabel("", fontsize=20)
    #fig.savefig('plots/dev_from_expected_breakdown.png', format='png', transparent=True, bbox_inches='tight')
    return None



def moca_vs_ic3_analysis(df_final):
    
    df_final = df_final.loc[df_final.user_id !='ic3study00062-session3-versionA',:] # subset moca scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00020-session1-versionA',:] # subset moca scores are missing
    df_final = df_final.loc[df_final.user_id != 'ic3study00052-session1-versionA',:] # subset moca scores are missing

    df_final = df_final.loc[df_final.user_id !='ic3study00076-session1-versionA',:] # ic3 scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00077-session1-versionA',:] # ic3 scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00078-session1-versionA',:] # ic3 scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00079-session1-versionA',:] # ic3 scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00083-session1-versionA',:] # ic3 scores are missing
    df_final = df_final.loc[df_final.user_id !='ic3study00108-session1-versionA',:] # ic3 scores are missing
    
    moca_tests = ['Moca total (30)','Trail Making (1)', 'visuospatial (4)','Naming (3)','Auditory Sustained Attention (1)','Arithmetic/Serial 7 (3)','Orientation (6)','Repetition (2)', 'Fluency (1)','Attention- digit span (2)','Abstraction (2)','Memory (5)']
    for test in moca_tests:
        df_final.loc[:,test] = df_final.loc[:,test].apply(lambda x: np.nan if x != x else int(x))
        
    df_final.loc[:,'visuospatial (4)'] = df_final.loc[:,'visuospatial (4)'].replace(5, 4)
    
    df_final.loc[:,'Visuospatial/Executive_moca'] = (df_final['visuospatial (4)'] + df_final['Trail Making (1)'])
    df_final.loc[:,'Language_moca'] = df_final['Naming (3)'] + df_final['Repetition (2)' ] + df_final['Fluency (1)'] + df_final['Abstraction (2)']
    df_final.loc[:,'Memory_moca'] = df_final['Memory (5)']
    df_final.loc[:,'Attention_moca'] = df_final['Auditory Sustained Attention (1)'] + df_final['Attention- digit span (2)']
    df_final.loc[:,'Orientation_moca'] = df_final['Orientation (6)']
    df_final.loc[:,'Calculation_moca'] = df_final['Arithmetic/Serial 7 (3)']
    df_final.loc[:,'auditoryAttention_moca'] = df_final['Auditory Sustained Attention (1)']
    df_final.loc[:,'trail_moca'] = df_final['Trail Making (1)']
    
    df_final.loc[:,'Visuospatial/Executive_moca'] = (df_final['Visuospatial/Executive_moca'] == np.max(df_final['Visuospatial/Executive_moca'])).astype(int)
    df_final.loc[:,'Language_moca'] = (df_final['Language_moca'] == np.max(df_final['Language_moca'])).astype(int)
    df_final.loc[:,'Memory_moca'] = (df_final['Memory_moca'] >= (np.max(df_final['Memory_moca'])-1)).astype(int)
    df_final.loc[:,'Attention_moca'] = (df_final['Attention_moca'] == np.max(df_final['Attention_moca'])).astype(int)
    df_final.loc[:,'Orientation_moca'] = (df_final['Orientation_moca'] == np.max(df_final['Orientation_moca'])).astype(int)
    df_final.loc[:,'Calculation_moca'] = (df_final['Calculation_moca'] == np.max(df_final['Calculation_moca'])).astype(int)
    df_final.loc[:,'auditoryAttention_moca'] = (df_final['auditoryAttention_moca'] == np.max(df_final['auditoryAttention_moca'])).astype(int)
    df_final.loc[:,'trail_moca'] = (df_final['trail_moca'] == np.max(df_final['trail_moca'])).astype(int)
    
    df_final.loc[:,'Naming'].replace(np.nan, 0, inplace=True)
    df_final.loc[:,'Reading'].replace(np.nan, 0, inplace=True)
    df_final.loc[:,'Repetition'].replace(np.nan, 0, inplace=True)
    swt_language = (df_final['Naming'] > -1.5) & (df_final['Repetition'] > -1.5) & (df_final['Reading'] > -1.5) 
    
    df_final['Visuospatial/Executive_ic3'] = ((df_final['Pear Cancellation'] >-1.5) & (df_final['Trail-making'] >-1.5) & (df_final['Rule Learning'] >-1.5) & (df_final['Odd One Out'] >-1.5) & (df_final['Blocks'] >-1.5)).astype(int)
    df_final['Language_ic3'] = ((df_final['Semantic Judgement'] >-1.5) & (df_final['Gesture Recognition'] >-1.5) & (df_final['Language Comprehension'] >-1.5) & swt_language).astype(int)
    df_final['Memory_ic3'] = ((df_final['Paired Associates Learning'] >-1.5) & (df_final['Task Recall'] >-1.5) & (df_final['Spatial Span'] >-1.5) & (df_final['Digit Span'] >-1.5) & (df_final['Orientation'] >-1.5)).astype(int) 
    df_final['Attention_ic3'] = ((df_final['Choice Reaction Time'] >-1.5) & (df_final['Simple Reaction Time'] >-1.5) & (df_final['Auditory Attention'] >-1.5)).astype(int)
    df_final['Orientation_ic3'] = ((df_final['Orientation'] >-1.5)).astype(int)
    df_final['Calculation_ic3'] = ((df_final['Graded Calculation'] >-1.5)).astype(int)
    df_final['auditoryAttention_ic3'] = ((df_final['Auditory Attention'] >-1.5)).astype(int)
    df_final['trail_ic3'] = ((df_final['Trail-making'] >-1.5)).astype(int)
    
    df_final.dropna(subset=['Moca total (30)'], inplace=True)
    df_final.drop_duplicates(subset=['ID'], inplace=True)
    
    domain_sensitivity = []
    domain_sensitivity.append(sum((df_final['Visuospatial/Executive_moca'] == 0) & (df_final['Visuospatial/Executive_ic3'] == 0))/len(df_final[df_final['Visuospatial/Executive_moca']==0]))
    domain_sensitivity.append(sum((df_final['Language_moca'] == 0) & (df_final['Language_ic3'] == 0))/len(df_final[df_final['Language_moca']==0]))
    domain_sensitivity.append(sum((df_final['Memory_moca'] == 0) & (df_final['Memory_ic3'] == 0))/len(df_final[df_final['Memory_moca']==0]))
    domain_sensitivity.append(sum((df_final['Attention_moca'] == 0) & (df_final['Attention_ic3'] == 0))/len(df_final[df_final['Attention_moca']==0]))
    domain_sensitivity.append(sum((df_final['Orientation_moca'] == 0) & (df_final['Orientation_ic3'] == 0))/len(df_final[df_final['Orientation_moca']==0]))
    domain_sensitivity.append(sum((df_final['Calculation_moca'] == 0) & (df_final['Calculation_ic3'] == 0))/len(df_final[df_final['Calculation_moca']==0]))
    domain_sensitivity.append(sum((df_final['auditoryAttention_moca'] == 0) & (df_final['auditoryAttention_ic3'] == 0))/len(df_final[df_final['auditoryAttention_moca']==0]))
    domain_sensitivity.append(sum((df_final['trail_moca'] == 0) & (df_final['trail_ic3'] == 0))/len(df_final[df_final['trail_moca']==0]))
    
    domain_sensitivity_ic3gold = []
    domain_sensitivity_ic3gold.append(sum((df_final['Visuospatial/Executive_moca'] == 0) & (df_final['Visuospatial/Executive_ic3'] == 0))/len(df_final[df_final['Visuospatial/Executive_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['Language_moca'] == 0) & (df_final['Language_ic3'] == 0))/len(df_final[df_final['Language_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['Memory_moca'] == 0) & (df_final['Memory_ic3'] == 0))/len(df_final[df_final['Memory_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['Attention_moca'] == 0) & (df_final['Attention_ic3'] == 0))/len(df_final[df_final['Attention_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['Orientation_moca'] == 0) & (df_final['Orientation_ic3'] == 0))/len(df_final[df_final['Orientation_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['Calculation_moca'] == 0) & (df_final['Calculation_ic3'] == 0))/len(df_final[df_final['Calculation_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['auditoryAttention_moca'] == 0) & (df_final['auditoryAttention_ic3'] == 0))/len(df_final[df_final['auditoryAttention_ic3']==0]))
    domain_sensitivity_ic3gold.append(sum((df_final['trail_moca'] == 0) & (df_final['trail_ic3'] == 0))/len(df_final[df_final['trail_ic3']==0]))
                        
    moca_comparison = pd.DataFrame(columns=['Domain','Sensitivity','Sensitivity_IC3gold'])
    moca_comparison.Domain = ['Visuospatial/Executive','Language','Memory','Attention','Orientation','Calculation','Auditory Attention', 'Trail-making']
    moca_comparison.Sensitivity = np.round(domain_sensitivity,4)
    moca_comparison.Sensitivity_IC3gold = np.round(domain_sensitivity_ic3gold,4)
    
    moca_comparison.Sensitivity = moca_comparison.Sensitivity.multiply(other=100)
    moca_comparison.Sensitivity_IC3gold = moca_comparison.Sensitivity_IC3gold.multiply(other=100)
        
    moca_comparison_long = pd.melt(moca_comparison, id_vars='Domain', value_vars=['Sensitivity','Sensitivity_IC3gold'])
    moca_comparison_long.replace({'variable':{'Sensitivity':'Sensitivity of IC3','Sensitivity_IC3gold':'Sensitivity of MOCA'}}, inplace=True)
    moca_comparison_long.rename(columns={'variable':'Legend'}, inplace=True)
    
    plot_moca_vs_ic3(moca_comparison_long)
    
    return None


def plot_moca_vs_ic3(moca_comparison_long):
    
    fig_dims=(30,20)
    sn.set_context("talk",font_scale=1.5)

    fig, ax = plt.subplots(figsize=fig_dims)

    sn.barplot(x='Domain',y='value', hue='Legend',palette=['#7851A9','#fff5cc'],
                data=moca_comparison_long, ax=ax)
    
    sn.set_style(style='ticks') 

    plt.text(1.95,105,'Domain-level Comparison',color='black',fontsize=26)
    plt.text(3.60,105,'Task-level Comparison',color='black',fontsize=26)
    ax.axvline(x=3.5,color='black',linestyle=(0, (5, 5)))
    ax.set_ylim(0, 115)
    
    # label each bar in barplot
    for p in ax.patches:
        # get the height of each bar
        height = p.get_height()
        # adding text to each bar
        ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar      
        y = height+0.55 if height>0 else height-0.4, # y-coordinate position of data label, padded 100 above bar
        s = "{:.0f}%".format(height) if height>0 else "{:.0f}\n*".format(height), # data label, formatted to ignore decimals
        ha = "center",
        fontsize= 20) # sets horizontal alignment (ha) to center
        
    ax.set(xlabel=None) 
    ax.set(ylabel="Percentage of impairments detected") 
    #plt.savefig('plots/sensitivity_IC3_and_moca.png', format='png', transparent=True, bbox_inches='tight')
    
    return None