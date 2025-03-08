"""
Updated on 3rd of May 2024
@authors: Dragos Gruia
"""

import os
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

def main_preprocessing(root_path, list_of_tasks, list_of_questionnaires, list_of_speech, remote_data_folders='/data_ic3online_cognition', supervised_data_folders=['/data_healthy_v1','/data_healthy_v2'],
                       folder_structure=['/summary_data','/trial_data','/speech'], output_clean_folder ='/data_healthy_cleaned', merged_data_folder ='/data_healthy_combined', clean_file_extension='_cleaned', data_format='.csv' ):

    print('Starting preprocessing...')
    
    os.chdir(root_path)

    print('Cleaning normative data.', end="", flush=True)
    print('Creating inclusion criteria...', end="", flush=True)
    
    ids_remote = general_outlier_detection_remoteSetting(root_path, remote_data_folders, folder_structure, screening_list = ['q_IC3_demographicsHealthy_questionnaire.csv', 'q_IC3_metacog_questionnaire.csv', 'IC3_Orientation.csv', 'IC3_PearCancellation.csv'])
    
    ids_supervised =  general_outlier_detection_supervisedSetting(root_path, supervised_data_folders, folder_structure, screening_list = ['q_IC3_demographics_questionnaire.csv', 'IC3_Orientation.csv', 'IC3_PearCancellation.csv'])
    
    inclusion_criteria = pd.concat([ids_remote, ids_supervised], ignore_index=True)
    inclusion_criteria.reset_index(drop=True,inplace=True)
    if not os.path.isdir(merged_data_folder[1:]):
        os.mkdir(merged_data_folder[1:])
        
    inclusion_criteria.to_csv(f'{root_path}{merged_data_folder}/inclusion_criteria.csv', index=False)
    print('Done')
    
    print('Merging data across sites...', end="", flush=True)
    
    list_of_tasks = merge_control_data_across_sites(root_path, folder_structure, supervised_data_folders, remote_data_folders, list_of_tasks,list_of_questionnaires,list_of_speech, data_format, merged_data_folder)

    print('Done')
    
    if list_of_tasks != None:
        for task_name in list_of_tasks:

            print(f'Pre-processing {task_name}...', end="", flush=True)
            df,df_raw = remove_general_outliers(root_path, merged_data_folder, task_name, inclusion_criteria,  data_format, folder_structure)

            match task_name:
                
                case 'IC3_Orientation':  
                    df,df_raw = orientation_preproc(df,df_raw)
                    
                case 'IC3_TaskRecall':  
                    df,df_raw = taskrecall_preproc(df,df_raw)
                    
                case 'IC3_rs_PAL':
                    df,df_raw = pal_preproc(df,df_raw)  

                case 'IC3_rs_digitSpan':
                    df,df_raw = digitspan_preproc(df,df_raw)

                case 'IC3_rs_spatialSpan':
                    df,df_raw = spatialspan_preproc(df,df_raw)                              
                
                case 'IC3_Comprehension':
                    df,df_raw = comprehension_preproc(df,df_raw)

                case 'IC3_SemanticJudgment':
                    df,df_raw = semantics_preproc(df,df_raw)
                    
                case 'IC3_BBCrs_blocks':
                    df,df_raw = blocks_preproc(df,df_raw)

                case 'IC3_NVtrailMaking':
                    df2,df_raw2 = remove_general_outliers(root_path, merged_data_folder, f'{task_name}2', inclusion_criteria, data_format, folder_structure)
                    df3,df_raw3 = remove_general_outliers(root_path, merged_data_folder, f'{task_name}3', inclusion_criteria, data_format, folder_structure)
                    df,df_raw = trailmaking_preproc(df,df2,df3,df_raw,df_raw2,df_raw3, task_name)
                    
                case 'IC3_rs_oddOneOut':
                    df,df_raw = oddoneout_preproc(df,df_raw)
                    
                case 'IC3_i4i_IDED':
                    df,df_raw = rule_learning_preproc(df,df_raw)
                    
                case 'IC3_PearCancellation':
                    df,df_raw = pear_cancellation_preproc(df,df_raw)
                
                case 'IC3_rs_SRT':
                    df,df_raw = srt_preproc(df,df_raw)
                    
                case 'IC3_AuditorySustainedAttention':
                    df,df_raw = auditory_attention_preproc(df,df_raw)
                    
                case 'IC3_rs_CRT':
                    df,df_raw = crt_preproc(df,df_raw)

                case 'IC3_i4i_motorControl':
                    df,df_raw = motor_control_preproc(df,df_raw)
                                
                case 'IC3_calculation':
                    df,df_raw = calculation_preproc(df,df_raw)
                    
                case 'IC3_GestureRecognition':
                    df,df_raw = gesture_preproc(df,df_raw)

                case _:
                    print(f'Task {task_name} does not have a specific preprocessing function.')
                    continue
            
            output_preprocessed_data(df,df_raw, root_path, output_clean_folder, folder_structure,  clean_file_extension, data_format)
            print('Done')
    else:
        print('No tasks were provided.') 
        
                   
    if list_of_questionnaires != None:
        for questionnaire_name in list_of_questionnaires:

            print(f'Pre-processing {questionnaire_name}...', end="", flush=True)
            
            match questionnaire_name:
                case 'q_IC3_demographics':
                    df_demographics = demographics_preproc(root_path, merged_data_folder, questionnaire_name, inclusion_criteria, folder_structure, data_format,clean_file_extension)
                    
                    combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure, list_of_tasks, df_demographics, clean_file_extension, data_format)
                    
                    print('Done')
                    
                case _:
                    print(f'Questionnaire {questionnaire_name} does not have a specific preprocessing function.')
                    

    else:
            print('No questionnaires were provided.')            
            
        
    print('Preprocessing complete.')      



def general_outlier_detection_remoteSetting(root_path, remote_data_folder, folder_structure, screening_list = ['q_IC3_demographicsHealthy_questionnaire.csv', 'q_IC3_metacog_questionnaire.csv', 'IC3_Orientation.csv', 'IC3_PearCancellation.csv']): 

    os.chdir(root_path + remote_data_folder + folder_structure[0])
        
    # Read the questionnaire data to check for exclusion criteria

    df_dem = pd.read_csv(screening_list[0], low_memory=False)
    df_cheat = pd.read_csv(screening_list[1], low_memory=False)
    df_orient = pd.read_csv(screening_list[2], low_memory=False)
    df_pear = pd.read_csv(screening_list[3], low_memory=False)

    # Remove duplicates

    df_dem.drop_duplicates(subset=['user_id'],keep="first", inplace=True)
    df_cheat.drop_duplicates(subset=['user_id'], keep="first", inplace=True)
    df_orient.drop_duplicates(subset=['user_id'],keep="first", inplace=True)
    df_pear.drop_duplicates(subset=['user_id'], keep="first", inplace=True)

    # Remove NAs

    df_dem.dropna(subset=['user_id'], inplace=True)
    df_cheat.dropna(subset=['user_id'], inplace=True)
    df_orient.dropna(subset=['user_id'], inplace=True)
    df_pear.dropna(subset=['user_id'], inplace=True)

    # Keep only the people who have done both the demographics and the screening tests.

    cleaned_ids_remote = list(set(df_pear['user_id']).intersection(set(df_pear['user_id']),set(df_dem['user_id'])))
    df_pear = df_pear[df_pear.user_id.isin(cleaned_ids_remote)].reset_index(drop=True)  
    df_orient = df_orient[df_orient.user_id.isin(cleaned_ids_remote)].reset_index(drop=True)  
    df_dem = df_dem[df_dem.user_id.isin(cleaned_ids_remote)].reset_index(drop=True)
    df_cheat = df_cheat[df_cheat.user_id.isin(cleaned_ids_remote)].reset_index(drop=True)  


    # Clean the questionnaire data

    df_dem.Q30_R.replace("No", "SKIPPED", inplace=True)
    df_dem.Q30_R.replace("SKIPPED", 999999, inplace=True)
    df_dem.Q30_R = df_dem.Q30_R.astype(float)

    df_cheat["Q2_S"].replace([0,1], np.nan, inplace=True)
    df_cheat["Q3_S"].replace([0,1], np.nan, inplace=True)
    df_cheat.dropna(subset=["Q2_S"], inplace=True)
    df_cheat.dropna(subset=["Q3_S"], inplace=True)
    df_cheat.reset_index(drop=True,inplace=True)

    # Drop the user_id, if they fail the screening test

    ids_failed_screen = []
    for subs in cleaned_ids_remote:
        if (df_orient[df_orient.user_id == subs].SummaryScore < 3).bool() and (df_pear[df_pear.user_id == subs].SummaryScore <= 0.80).bool():
            
            ids_failed_screen.append(subs)
            df_dem = df_dem.drop(df_dem[df_dem.user_id == subs].index).reset_index(drop=True)
            df_orient = df_orient.drop(df_orient[df_orient.user_id == subs].index).reset_index(drop=True)
            df_pear = df_pear.drop(df_pear[df_pear.user_id == subs].index).reset_index(drop=True)

    print(f'We removed {len(ids_failed_screen)} people who failed both Orientation and Pear Cancellation, from all tasks.')


    # Update the user_ids list to only include the people who passed the screening test

    cleaned_ids_remote = df_pear.user_id

    # Remove people who are not neurologically healthy

    to_remove = (df_dem.Q12_R != "SKIPPED") | (df_dem.Q14_R != "SKIPPED") | (df_dem.Q30_R <= 60) | (df_dem.Q1_R < 40)    
    print(f'We removed {sum((df_dem.Q12_R != "SKIPPED") | (df_dem.Q14_R != "SKIPPED"))} who indicated they have a neurological disorder, {sum(df_dem.Q30_R <= 60)} who have a history of dementia and {sum(df_dem.Q1_R < 40)} who are younger than 40.')

    df_dem = df_dem[~to_remove].reset_index(drop=True)
    df_pear = df_pear[~to_remove].reset_index(drop=True)
    df_orient = df_orient[~to_remove].reset_index(drop=True)

    cleaned_ids_remote = df_pear.user_id


    # Remove people who self-reeported lack of engagement
    cheating_ids = cleaned_ids_remote.isin(df_cheat.user_id)
    df_dem = df_dem[~cheating_ids].reset_index(drop=True)
    df_pear = df_pear[~cheating_ids].reset_index(drop=True)
    df_orient = df_orient[~cheating_ids].reset_index(drop=True)

    print(f'We removed {cheating_ids.sum()} people who cheated.')
    cleaned_ids_remote = df_pear.user_id


    return cleaned_ids_remote # Return participant ids that passed exclusion criteria


def general_outlier_detection_supervisedSetting(root_path, supervised_data_folders, folder_structure, screening_list = ['q_IC3_demographics_questionnaire.csv', 'IC3_Orientation.csv', 'IC3_PearCancellation.csv']): 

    # Read the data for v1

    os.chdir(root_path + supervised_data_folders[0] + folder_structure[0])

    df_dem_tp1 = pd.read_csv(screening_list[0], low_memory=False)
    df_orient_tp1 = pd.read_csv(screening_list[1], low_memory=False)
    df_pear_tp1 = pd.read_csv(screening_list[2], low_memory=False)

    # Read the data for v2

    os.chdir(root_path + supervised_data_folders[1] + folder_structure[0])

    df_dem_tp2 = pd.read_csv(screening_list[0], low_memory=False)
    df_orient_tp2 = pd.read_csv(screening_list[1], low_memory=False)
    df_pear_tp2 = pd.read_csv(screening_list[2], low_memory=False)

    # Concatenate the two timepoints

    df_dem = pd.concat([df_dem_tp1, df_dem_tp2], ignore_index=True)
    df_orient = pd.concat([df_orient_tp1, df_orient_tp2], ignore_index=True)
    df_pear = pd.concat([df_pear_tp1, df_pear_tp2], ignore_index=True)

    # Remove duplicates, NAs and reset index

    df_dem.drop_duplicates(subset=['user_id'],keep="first", inplace=True)
    df_dem.dropna(subset=['user_id'], inplace=True)
    df_dem.reset_index(drop=True, inplace=True)

    df_orient.drop_duplicates(subset=['user_id'],keep="first", inplace=True)
    df_orient.dropna(subset=['user_id'], inplace=True)
    df_orient.reset_index(drop=True, inplace=True)

    df_pear.drop_duplicates(subset=['user_id'],keep="first", inplace=True)
    df_pear.dropna(subset=['user_id'], inplace=True)
    df_pear.reset_index(drop=True, inplace=True)

    # Find the user_id who are in both df_pear and df_pear

    cleaned_ids_supervised = list(set(df_dem['user_id']).intersection(set(df_orient['user_id'])).intersection(set(df_pear['user_id'])))

    df_pear = df_pear[df_pear.user_id.isin(cleaned_ids_supervised)]  
    df_orient = df_orient[df_orient.user_id.isin(cleaned_ids_supervised)]  
    df_dem = df_dem[df_dem.user_id.isin(cleaned_ids_supervised)]  

    # Drop the user_id, if they fail the screening test

    ids_failed_screen = []
    for subs in cleaned_ids_supervised:
        if (df_orient[df_orient.user_id == subs].SummaryScore < 3).bool() and (df_pear[df_pear.user_id == subs].SummaryScore <= 0.80).bool():
            
            ids_failed_screen.append(subs)
            df_dem = df_dem.drop(df_dem[df_dem.user_id == subs].index).reset_index(drop=True)
            df_orient = df_orient.drop(df_orient[df_orient.user_id == subs].index).reset_index(drop=True)
            df_pear = df_pear.drop(df_pear[df_pear.user_id == subs].index).reset_index(drop=True)

    print(f'We removed {len(ids_failed_screen)} people who failed both Orientation and Pear Cancellation, from all tasks.')


    cleaned_ids_supervised = df_pear.user_id

    # Remove people who are not neurologically healthy

    to_remove = (df_dem.Q12_R != "SKIPPED") | (df_dem.Q14_R != "SKIPPED") | (df_dem.Q1_R < 40)    
    print(f'We removed {sum((df_dem.Q12_R != "SKIPPED") | (df_dem.Q14_R != "SKIPPED"))} who indicated they have a neurological disorder, and {sum(df_dem.Q1_R < 40)} who are younger than 40.')

    df_dem = df_dem[~to_remove].reset_index(drop=True)
    df_pear = df_pear[~to_remove].reset_index(drop=True)
    df_orient = df_orient[~to_remove].reset_index(drop=True)

    cleaned_ids_supervised = df_pear.user_id

    return cleaned_ids_supervised # Return participant ids that passed exclusion criteria



def merge_control_data_across_sites(root_path, folder_structure, supervised_data_folders, remote_data_folders, list_of_tasks,list_of_questionnaires,list_of_speech, data_format, merged_data_folder):
        
    os.chdir(root_path)

    # Create folder structure
    
    if not os.path.isdir(merged_data_folder[1:]):
        os.mkdir(merged_data_folder[1:])
    os.chdir(merged_data_folder[1:])

    if not os.path.isdir(folder_structure[0][1:]):
        os.mkdir(folder_structure[0][1:])
        
    if not os.path.isdir(folder_structure[1][1:]):
        os.mkdir(folder_structure[1][1:])
    
    if not os.path.isdir(folder_structure[2][1:]):
        os.mkdir(folder_structure[2][1:])

    # Merge data from clinical tests
    
    if 'IC3_NVtrailMaking' in list_of_tasks:
        list_of_tasks.append('IC3_NVtrailMaking2')
        list_of_tasks.append('IC3_NVtrailMaking3')

    for taskName in list_of_tasks:
        
        summary_task_path = root_path + supervised_data_folders[0] + folder_structure[0]
        raw_task_path = root_path + supervised_data_folders[0] + folder_structure[1]
        
        df_v1 = pd.read_csv(f'{summary_task_path}/{taskName}{data_format}', low_memory=False)
        df_v1_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw{data_format}'), low_memory=False)
        
        summary_task_path = root_path + supervised_data_folders[1] + folder_structure[0]
        raw_task_path = root_path + supervised_data_folders[1] + folder_structure[1]
        
        # Special case for IDED task that has two versions
        
        if taskName == "IC3_i4i_IDED": 
            df_v2 = pd.read_csv((f'{summary_task_path}/{taskName}2{data_format}'), low_memory=False)
            df_v2_raw = pd.read_csv((f'{raw_task_path}/{taskName}2_raw{data_format}'), low_memory=False)
        else:   
            df_v2 = pd.read_csv(f'{summary_task_path}/{taskName}{data_format}', low_memory=False)
            df_v2_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw{data_format}'), low_memory=False)
        
        
        summary_task_path = root_path + remote_data_folders + folder_structure[0]
        raw_task_path = root_path + remote_data_folders + folder_structure[1]

        df_cog = pd.read_csv(f'{summary_task_path}/{taskName}{data_format}', low_memory=False)
        df_cog_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw{data_format}'), low_memory=False)
        
        df = pd.concat([df_v1, df_v2, df_cog], ignore_index=True)
        df_raw = pd.concat([df_v1_raw, df_v2_raw, df_cog_raw], ignore_index=True)
        
        output_folder = root_path + merged_data_folder
        
        df.to_csv(f'{output_folder}/{folder_structure[0]}/{taskName}{data_format}', index=False)
        df_raw.to_csv(f'{output_folder}/{folder_structure[1]}/{taskName}_raw{data_format}', index=False)
        #print(f'Merged {taskName}')
        
    if 'IC3_NVtrailMaking' in list_of_tasks:
        list_of_tasks = list_of_tasks[:-2]
        
    # Merge data from speech
        
    for taskName in list_of_speech:
        
        raw_speech_path = root_path + supervised_data_folders[0] + folder_structure[1]
        df_v1_raw = pd.read_csv((f'{raw_speech_path}/{taskName}_raw{data_format}'), low_memory=False)
        
        raw_speech_path = root_path + supervised_data_folders[1] + folder_structure[1]
        df_v2_raw = pd.read_csv((f'{raw_speech_path}/{taskName}_raw{data_format}'), low_memory=False)

        df_raw = pd.concat([df_v1_raw, df_v2_raw], ignore_index=True)
        
        output_folder = root_path + merged_data_folder
        df_raw.to_csv(f'{output_folder}/{folder_structure[1]}/{taskName}_raw.csv', index=False)
        #print(f'Merged {taskName}')
        
    # Merge data from questionnaires

    for taskName in list_of_questionnaires:
        
        if taskName == 'q_IC3_demographics':
            
            summary_task_path = root_path + supervised_data_folders[0] + folder_structure[0]
            raw_task_path = root_path + supervised_data_folders[0] + folder_structure[1]
            
            df_v1 = pd.read_csv((f'{summary_task_path}/{taskName}Healthy_questionnaire.csv'), low_memory=False)
            df_v1_2 = pd.read_csv((f'{summary_task_path}/{taskName}_questionnaire.csv'), low_memory=False)
            df_v1_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw.csv'), low_memory=False)
            df_v1_raw_2 = pd.read_csv((f'{raw_task_path}/{taskName}Healthy_raw.csv'), low_memory=False)

            summary_task_path = root_path + supervised_data_folders[1] + folder_structure[0]
            raw_task_path = root_path + supervised_data_folders[1] + folder_structure[1]
 
            df_v2 = pd.read_csv((f'{summary_task_path}/{taskName}_questionnaire.csv'), low_memory=False)
            df_v2_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw.csv'), low_memory=False)
            
            summary_task_path = root_path + remote_data_folders + folder_structure[0]
            raw_task_path = root_path + remote_data_folders + folder_structure[1]

            df_cog = pd.read_csv((f'{summary_task_path}/{taskName}Healthy_questionnaire.csv'), low_memory=False)
            df_cog_raw = pd.read_csv((f'{raw_task_path}/{taskName}Healthy_raw.csv'), low_memory=False)

            df = pd.concat([df_v1, df_v1_2, df_v2, df_cog], ignore_index=True)
            df_raw = pd.concat([df_v1_raw, df_v1_raw_2, df_v2_raw, df_cog_raw], ignore_index=True)
            
        else:
            
            summary_task_path = root_path + supervised_data_folders[0] + folder_structure[0]
            raw_task_path = root_path + supervised_data_folders[0] + folder_structure[1]
            
            df_v1 = pd.read_csv(f'{summary_task_path}/{taskName}_questionnaire.csv', low_memory=False)
            df_v1_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw.csv'), low_memory=False)
            
            summary_task_path = root_path + supervised_data_folders[1] + folder_structure[0]
            raw_task_path = root_path + supervised_data_folders[1] + folder_structure[1]
            
            df_v2 = pd.read_csv(f'{summary_task_path}/{taskName}_questionnaire.csv', low_memory=False)
            df_v2_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw.csv'), low_memory=False)
            
            summary_task_path = root_path + remote_data_folders + folder_structure[0]
            raw_task_path = root_path + remote_data_folders + folder_structure[1]
            
            df_cog = pd.read_csv(f'{summary_task_path}/{taskName}_questionnaire.csv', low_memory=False)
            df_cog_raw = pd.read_csv((f'{raw_task_path}/{taskName}_raw.csv'), low_memory=False)
            
            df = pd.concat([df_v1, df_v2, df_cog], ignore_index=True)
            df_raw = pd.concat([df_v1_raw, df_v2_raw, df_cog_raw], ignore_index=True)
        
        output_folder = root_path + merged_data_folder

        df.to_csv(f'{output_folder}/{folder_structure[0]}/{taskName}{data_format}', index=False)
        df_raw.to_csv(f'{output_folder}/{folder_structure[1]}/{taskName}_raw.csv', index=False)
        #print(f'Merged {taskName}')
    
    return list_of_tasks


def combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure, list_of_tasks, df_demographics, clean_file_extension, data_format):
    
    os.chdir(root_path + output_clean_folder + folder_structure[0])
    
    #Do a loop and read the data in each of the tasks

    for file in list_of_tasks:
        
        # Read Task Data for Healthy Participants
        temp_healthy_cog = pd.read_csv(f'{file}{clean_file_extension}{data_format}', low_memory=False)
        
        # Drop duplicates if any
        temp_healthy_cog.drop_duplicates(subset='user_id', keep='last', inplace=True)

        # Reset the Index
        temp_healthy_cog.reset_index(drop=True, inplace=True)

        # Merge Demographics data between patients and controls
        task_id = temp_healthy_cog.taskID.iloc[0]       
        temp_healthy_cog = temp_healthy_cog.loc[:,['user_id','SummaryScore']]
        temp_healthy_cog.rename(columns={'SummaryScore':task_id}, inplace=True)
        
        df_demographics= pd.merge(df_demographics, temp_healthy_cog,  on="user_id", how="left")
    
    df_demographics.to_csv('summary_cognition_and_demographics.csv')
    return None

        
    

def demographics_preproc(root_path, merged_data_folder, questionnaire_name, inclusion_criteria, folder_structure, data_format, clean_file_extension):
    
    os.chdir(root_path + merged_data_folder)
    try:
        df_dem_summary = pd.read_csv(f'.{folder_structure[0]}/{questionnaire_name}{data_format}', low_memory=False)
        df_dem_summary = df_dem_summary[df_dem_summary.user_id.isin(inclusion_criteria)]
        
        df_dem = pd.read_csv(f'.{folder_structure[1]}/{questionnaire_name}_raw{data_format}', low_memory=False)
        df_dem = df_dem[df_dem.user_id.isin(inclusion_criteria)]    

    except:
        print(f'Error in loading {questionnaire_name}. File might not exist.')
        return None
    
    df_dem.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_dem =df_dem.drop_duplicates(subset=['user_id','question'], keep='last').reset_index(drop=True)
    df_dem_summary =df_dem_summary.drop_duplicates(subset='user_id',keep='last').reset_index(drop=True)
    
    # Extract demographics of interest
    
    age = df_dem.groupby(['question']).get_group('<center>Howoldareyou?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    gender = df_dem.groupby(['question']).get_group('<center>Whatisyourbiologicalsex?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    education = df_dem.groupby(['question']).get_group('<center>Whatisyourhighestlevelofeducation?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    device = df_dem.groupby(['question']).get_group('<center>Whatdeviceareyouusingatthemoment?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    english = df_dem.groupby(['question']).get_group('<center>HowwouldyourateyourproficiencyinEnglish?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    depression = df_dem.groupby(['question']).get_group('<center>Areyoucurrentlytakinganymedicationfordepression?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    anxiety = df_dem.groupby(['question']).get_group('<center>Areyoucurrentlytakinganymedicationforanxiety?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    dyslexia = df_dem.groupby(['question']).get_group('<center>DoyouhaveDyslexia,oranyotherproblemswithreadingandwriting?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    risks = df_dem.groupby(['question']).get_group('<center>Haveyoueverbeentoldyouhavethefollowing?Tickallthatapplies</center>').loc[:,['response','user_id']].reset_index(drop=True)

    # Clean each variable
    
    age.response = pd.to_numeric(age.response)
    age.loc[age.response < 40,'response'] = np.nan
    
    gender.response = gender.response.replace(['53','55','65','78','71','72'],np.nan)
    gender.replace(['Male','Female'],[0,1], inplace=True)

    education.response = education.response.replace('1',np.nan)
    education.response = education.response.replace('Secondary/HighSchoolDiploma','Secondary/HighSchool-A-levels')
    education.response = education.response.replace('Primary/ElementarySchool','SecondarySchool-GCSE')
    education.replace(['SecondarySchool-GCSE','Secondary/HighSchool-A-levels','ProfessionalDegree','Bachelor\'sDegree','Master\'sDegree','PhD'],[0,1,1,2,3,3], inplace=True)
                
    device = device.merge(df_dem_summary.loc[:,['user_id','os']], on='user_id', how='outer')
    device.response = device.response.fillna(device.os)
    device.drop('os',axis=1,inplace=True)
    
    english.response.replace({'3': 1, '4': 1, 'No': np.nan, '2':1,'1':0}, inplace=True)
    depression.response.replace({'No':0, 'Yes':1, 'SKIPPED':0}, inplace=True)   
    anxiety.response.replace({'No':0, 'Yes':1, 'SKIPPED':0}, inplace=True)
    dyslexia.response.replace({'Yes':1,'No':0,'Tablet':0,'Touchscreen':0,np.nan:0}, inplace=True)
    
    risks.drop_duplicates(keep='last', inplace=True)
    risks.replace(np.nan, ' ', inplace=True)
    risks.response = risks.response.str.lower()
    risks['diabetes'] = risks.response.apply(lambda x: 'diabetes' in x).replace([True,False], [1,0])
    risks['highbloodpressure'] = risks.response.apply(lambda x: 'highbloodpressure' in x).replace([True,False], [1,0])
    risks['highcholesterol'] = risks.response.apply(lambda x: ('highcholesterol' in x) or ('highcholesterole' in x)).replace([True,False], [1,0])
    risks['heartdisease'] = risks.response.apply(lambda x: 'heartdisease' in x).replace([True,False], [1,0])
    risks['kidneydisease'] = risks.response.apply(lambda x: 'kidneydisease' in x).replace([True,False], [1,0])
    risks['alcoholdependency'] = risks.response.apply(lambda x: 'alcoholdependency' in x).replace([True,False], [1,0])
    risks['over-weight'] = risks.response.apply(lambda x: ('over-weight' in x) or ('overweight' in x)).replace([True,False], [1,0])
    risks['long-termsmoker'] = risks.response.apply(lambda x: ('long-termsmoker' in x) or ('longtermsmoker' in x)).replace([True,False], [1,0])
    risks['ex-smoker'] = risks.response.apply(lambda x: ('ex-smoker' in x) or ('exsmoker' in x)).replace([True,False], [1,0])
    risks.loc[(risks['long-termsmoker'] & risks['ex-smoker']).astype(bool),'ex-smoker'] = 0
    risks.response = risks.iloc[:,2:].sum(axis=1)
    
    age.drop_duplicates(subset="user_id",keep='last',inplace=True)
    gender.drop_duplicates(subset="user_id", keep='last',inplace=True)
    education.drop_duplicates(subset="user_id", keep='last',inplace=True)
    device.drop_duplicates(subset="user_id", keep='last',inplace=True)
    english.drop_duplicates(subset="user_id", keep='last',inplace=True)
    depression.drop_duplicates(subset="user_id", keep='last',inplace=True)
    anxiety.drop_duplicates(subset="user_id", keep='last',inplace=True)
    dyslexia.drop_duplicates(subset="user_id", keep='last',inplace=True)
    risks.drop_duplicates(subset="user_id", keep='last',inplace=True)
    
    age.rename(columns={'response':'age'}, inplace=True)
    gender.rename(columns={'response':'gender'}, inplace=True)
    education.rename(columns={'response':'education'}, inplace=True)
    device.rename(columns={'response':'device'}, inplace=True)
    english.rename(columns={'response':'english'}, inplace=True)
    depression.rename(columns={'response':'depression'}, inplace=True)
    anxiety.rename(columns={'response':'anxiety'}, inplace=True)
    dyslexia.rename(columns={'response':'dyslexia'}, inplace=True)
    risks.rename(columns={'response':'risks'}, inplace=True)
    
    age.dropna(inplace=True)
    gender.dropna(inplace=True)
    education.dropna(inplace=True)
    device.dropna(inplace=True)
    english.dropna(inplace=True)
    depression.dropna(inplace=True)
    anxiety.dropna(inplace=True)
    dyslexia.dropna(inplace=True)
    risks.dropna(inplace=True)
              
    # Merge and format
    
    healthy_demographics = age.merge(gender,on='user_id').merge(education,on='user_id').merge(device,on='user_id').merge(english,on='user_id').merge(depression,on='user_id').merge(anxiety,on='user_id').merge(risks,on='user_id').merge(dyslexia,on='user_id')
    healthy_demographics.education = healthy_demographics.education.astype(int)
    
    one_hot_encoded_data = pd.get_dummies(healthy_demographics, columns = ['device', 'education'])
    one_hot_encoded_data.rename(columns={'education_1':'education_Alevels', 'education_2':'education_bachelors','education_3':'education_postBachelors'}, inplace=True)
    one_hot_encoded_data.rename(columns={'device_1':'device_tablet', 'device_0':'device_phone'}, inplace=True)
    one_hot_encoded_data.rename(columns={'english':'english_secondLanguage'}, inplace=True)
    one_hot_encoded_data.replace({True:1, False:0}, inplace=True)
    one_hot_encoded_data.loc[:,'gender':'education_postBachelors'] = one_hot_encoded_data.loc[:,'gender':'education_postBachelors'] -0.5

    # Save 
    
    one_hot_encoded_data.to_csv(f'.{folder_structure[0]}/{questionnaire_name}{clean_file_extension}{data_format}', index=False)
    
    return one_hot_encoded_data

  
def remove_general_outliers(root_path, merged_data_folder, task_name, inclusion_criteria,  data_format, folder_structure=['/summary_data','/trial_data','/speech']):
    
    path_to_data = root_path + merged_data_folder
    
    #try:
    df = pd.read_csv(f'{path_to_data}/{folder_structure[0]}/{task_name}{data_format}', low_memory=False)
    df = df[df.user_id.isin(inclusion_criteria)]
    
    df_raw = pd.read_csv(f'{path_to_data}/{folder_structure[1]}/{task_name}_raw{data_format}', low_memory=False)
    df_raw = df_raw[df_raw.user_id.isin(inclusion_criteria)]
    #except:
    #    print(f'Error in loading {task_name}. File might not exist.')
    #    return None,None

    df.drop_duplicates(subset=['user_id'],keep="last", inplace=True)
    df.drop(columns=['Unnamed: 0','Level','type','RespObject','sequenceObj', 'dynamicDifficulty'], inplace=True)
    df.reset_index(drop=True,inplace=True)

    if ('Unnamed: 0' in df_raw.columns):
        
        df_raw = df_raw.rename(columns={'Unnamed: 0':'Level_filter'})

    df_raw.drop_duplicates(subset=['user_id','Level_filter'],keep="last", inplace=True)
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw

def output_preprocessed_data(df,df_raw, root_path, output_clean_folder, folder_structure, clean_file_extension, data_format):
    
    os.chdir(root_path)
    
    if os.path.isdir(output_clean_folder[1:]) == False:
        os.mkdir(output_clean_folder[1:])
        os.mkdir(f'.{output_clean_folder}{folder_structure[0]}')
        os.mkdir(f'.{output_clean_folder}{folder_structure[1]}')

    df.to_csv(f".{output_clean_folder}{folder_structure[0]}/{df.loc[0,'taskID']}{clean_file_extension}{data_format}", index=False)
    df_raw.to_csv(f".{output_clean_folder}{folder_structure[1]}/{df.loc[0,'taskID']}_raw{clean_file_extension}{data_format}")

    return None

def orientation_preproc(df,df_raw):

    df_raw.loc[df_raw.RT<400,'RT'] = np.nan
    df_raw.loc[df_raw.RT>10000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        date_raw = df_raw_temp.iloc[0].TimeRespEnabled
        datestart = datetime.datetime.fromtimestamp(float(date_raw)/1000).strftime('%Y-%m-%d %H:%M:%S')
        datestart = datestart.split(' ')[1].split(':')
        
        if any(x == int(datestart[0]) for x in [6, 12, 18, 0]) and (int(datestart[1]) < 30):
            df_raw_temp.loc[df_raw_temp.Level == 4, 'correct'] = True
        elif any(x == int(datestart[0]) for x in [5, 11, 17, 23]) and (int(datestart[1]) > 30):
            df_raw_temp.loc[df_raw_temp.Level == 4, 'correct'] = True
            
        errors[count] = (sum(df_raw_temp.correct == False))
        scores[count] = sum(df_raw_temp.correct)
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
    df_temp = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})

    df["SummaryScore"] = df_temp.score
    df["totalCorrect"] = df_temp.score
    df["errors"] = df_temp.errors
    df["medianRT"] = df_temp.medianRT
    df["meanRT"] = df_temp.meanRT
    df["medianCorrectRT"] = df_temp.medianCorrRT
    df["medianErrorRT"] = df_temp.medianErrorRT
    df["meanCorrectRT"] = df_temp.meanCorrRT
    df["meanErrorRT"] = df_temp.meanErrorRT

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df_temp.numNAs > 0.6) | (df.SummaryScore < 2)) & (df.SummaryScore <4)
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw

def pear_cancellation_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>12000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    shortRTs = [None] * len(df.user_id)
    ids = [None] * len(df.user_id)
    leftPear = [None] * len(df.user_id)
    rightPear = [None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        df_raw_temp = df_raw[df_raw.user_id == id]
        ids[count] = id
        df_raw_temp = df_raw_temp[(df_raw_temp.PearNumber > 0)]
        df_raw_temp.drop_duplicates(subset=['Response'],keep="first", inplace=True)
        
        errors[count] = (sum(df_raw_temp.correct == False))
        leftPear[count] = max(df_raw_temp.PearLeft) if not (df_raw_temp.PearLeft).empty else np.nan
        rightPear[count] = max(df_raw_temp.PearRight) if not (df_raw_temp.PearRight).empty else np.nan
        scores[count] = max(df_raw_temp.PearFull) if not (df_raw_temp.PearFull).empty else np.nan

        shortRTs[count] = sum(df_raw_temp.RT < 1000)
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
    df_temp = pd.DataFrame({"user_id":ids, "score":scores, "errors":errors, "meanRT":meanRTs, "shortRTs":shortRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "leftPearClicked":leftPear, "rightPearClicked":rightPear, "numNAs":numNAs})
        
    df["SummaryScore"] = df_temp.score/20
    df["prcCorrectPears"] = df_temp.score/20
    df["errors"] = df_temp.errors
    df["medianRT"] = df_temp.medianRT
    df["meanRT"] = df_temp.meanRT
    df["medianCorrectRT"] = df_temp.medianCorrRT
    df["medianErrorRT"] = df_temp.medianErrorRT
    df["meanCorrectRT"] = df_temp.meanCorrRT
    df["meanErrorRT"] = df_temp.meanErrorRT
    df["leftPearClicked"] = df_temp.leftPearClicked
    df["rightPearClicked"] = df_temp.rightPearClicked    

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore < 0.8)) & (df.SummaryScore <0.90)
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw

def digitspan_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>30000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    ids = [None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        ids[count] = id
        df_raw_temp.drop_duplicates(subset=['Response','RT'],keep="first", inplace=True)
        
        errors[count] = (sum(df_raw_temp.correct == False))
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan  
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        df_raw_temp.drop(df_raw_temp[df_raw_temp.correct == False].index, inplace=True)
        df_raw_temp.reset_index(drop=True,inplace=True)
        
        scores[count] = len((df_raw_temp.Response.loc[len(df_raw_temp)-1]).split('_'))-1 if not (df_raw_temp.Response).empty else 0
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":ids, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})

    df["SummaryScore"] = df_raw_2.score
    df["maxAchieved"] = df_raw_2.score
    df["errors"] = df_raw_2.errors
    df["medianRT"] = df_raw_2.medianRT
    df["meanRT"] = df_raw_2.meanRT
    df["medianCorrectRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanCorrectRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    
    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore < 2))
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw    


def spatialspan_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>30000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    ids = [None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        ids[count] = id
        df_raw_temp.drop_duplicates(subset=['Response','RT'],keep="first", inplace=True)
        
        errors[count] = (sum(df_raw_temp.correct == False))
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan  
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        df_raw_temp.drop(df_raw_temp[df_raw_temp.correct == False].index, inplace=True)
        df_raw_temp.reset_index(drop=True,inplace=True)
        
        scores[count] = len((df_raw_temp.Response.loc[len(df_raw_temp)-1]).split('_'))-1 if not (df_raw_temp.Response).empty else 0
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":ids, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})

    df["SummaryScore"] = df_raw_2.score
    df["maxAchieved"] = df_raw_2.score
    df["errors"] = df_raw_2.errors
    df["medianRT"] = df_raw_2.medianRT
    df["meanRT"] = df_raw_2.meanRT
    df["medianCorrectRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanCorrectRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
        
    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore < 3) | (df.SummaryScore > 10))
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw  



def pal_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>30000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    ids = [None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        ids[count] = id        
        
        errors[count] = (sum(df_raw_temp.correct == False))
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan  
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        df_raw_temp.drop(df_raw_temp[df_raw_temp.correct == False].index, inplace=True)
        df_raw_temp.reset_index(drop=True,inplace=True)
        
        temp2 = df_raw_temp[df_raw_temp.Level == df_raw_temp["Cue Number"] ]
        scores[count] = df_raw_temp[(df_raw_temp["Level"] == df_raw_temp["Cue Number"]) & df_raw_temp.correct == True].Level.max() if (not (df_raw_temp.correct).empty) & (df_raw_temp.Level.min() <= df_raw_temp["Cue Number"].max()) else 0
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":ids, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})

    df["SummaryScore"] = df_raw_2.score
    df["maxAchieved"] = df_raw_2.score
    df["totalErrors"] = df_raw_2.errors
    df["medianRT"] = df_raw_2.medianRT
    df["meanRT"] = df_raw_2.meanRT
    df["medianCorrectRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanCorrectRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
  
    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore < 2) | (df.SummaryScore >9))
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw  



def semantics_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<500,'RT'] = np.nan
    df_raw.loc[df_raw.RT>30000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    shortRTs = [None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)
    for count,id in enumerate(df.user_id):
        df_raw_temp = df_raw[df_raw.user_id == id]
        errors[count] = (sum(df_raw_temp.correct == False))
        temp_score = df_raw_temp.groupby("Level")["correct"].sum()
        for x in range (len(temp_score)-1):
            if temp_score[x] >3:
                temp_score[x] = 3
        scores[count]= sum(temp_score)
        shortRTs[count]= sum(df_raw_temp.RT < 1000)
        meanRTs[count]= np.nanmean(df_raw_temp.RT)
        medianRTs[count]= np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count]= np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "shortRTs":shortRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "numNAs":numNAs})
    df["SummaryScore"] = df_raw_2.score
    df["totalCorrect"] = df_raw_2.score
    df["totalSkipped"] = df_raw_2.errors
    df["medianRT"] = df_raw_2.medianRT
    df["meanRT"] = df_raw_2.meanRT
    df["medianCorrectRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore <= 15))  
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw  


def srt_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<180,'RT'] = np.nan
    df_raw.loc[df_raw.RT>800,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    ids = [None] * len(df.user_id)
    numMisclicks = [None]* len(df.user_id)
    numTimeOuts = [None]* len(df.user_id)
    numCorrectClicks = [None]* len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        ids[count] = id        
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        numMisclicks[count] = np.sum(df_raw_temp.Misclick)
        numTimeOuts[count] = np.sum(df_raw_temp['Time Out'])
        numCorrectClicks[count] = np.sum((df_raw_temp.Misclick == 0) & (df_raw_temp['Time Out'] == 0))
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
    df_raw_2 = pd.DataFrame({"user_id":ids, "meanRT":meanRTs, "medianRT":medianRTs, "numMisclicks":numMisclicks, "numTimeOuts":numTimeOuts, "numCorrectClicks":numCorrectClicks, "numNAs":numNAs})
    df["SummaryScore"] = df_raw_2.meanRT
    df["medianRT"] = df_raw_2.medianRT
    df["meanRT"] = df_raw_2.meanRT
    df["numMisclicks"] = df_raw_2.numMisclicks
    df["numTimeOuts"] = df_raw_2.numTimeOuts
    df["numCorrectClicks"] = df_raw_2.numCorrectClicks

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.numCorrectClicks < 30) | (df.numTimeOuts > 14) | (df.numMisclicks > 14))    
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    

    return df,df_raw  


def crt_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>1000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    numTimeOuts = [None]* len(df.user_id)
    numMisclicks = [None]* len(df.user_id)
    numIncorr = [None]* len(df.user_id)
    numCorrectClicks = [None]* len(df.user_id)
    numberNAs = [None]* len(df.user_id)

    for count,id in enumerate(df.user_id):
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        meanRTs[count]= np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count]= np.nanmean(df_raw_temp[df_raw_temp.Correct == 'false'].RT) if (df_raw_temp.Correct == 'false').any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.Correct == 'true'].RT) if (df_raw_temp.Correct == 'true').any() else np.nan

        medianRTs[count]= np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count]= np.nanmedian(df_raw_temp[df_raw_temp.Correct == 'false'].RT) if (df_raw_temp.Correct == 'false').any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.Correct == 'true'].RT) if (df_raw_temp.Correct == 'true').any() else np.nan
        
        numberNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
        numTimeOuts[count] = np.sum(df_raw_temp.Correct == 'TIMEOUT')  if (df_raw_temp.Correct == 'TIMEOUT').any() else 0
        numMisclicks[count] = np.sum(df_raw_temp.Missclick)
        numIncorr[count] = np.sum(df_raw_temp.Correct == 'false')  if (df_raw_temp.Correct == 'false').any() else 0
        numCorrectClicks[count] = np.sum(df_raw_temp.Correct == 'true')  if (df_raw_temp.Correct == 'true').any() else 0
        
    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numTimeOuts":numTimeOuts, "numMisclicks":numMisclicks, "numIncorr":numIncorr, "numCorrectClicks":numCorrectClicks, "numberNAs":numberNAs})
    df["SummaryScore"] = df_raw_2.numCorrectClicks
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["numTimeOuts"] = df_raw_2.numTimeOuts
    df["numMisclicks"] = df_raw_2.numMisclicks
    df["numIncorr"] = df_raw_2.numIncorr
    df["numCorrectClicks"] = df_raw_2.numCorrectClicks

    exc = (df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.numCorrectClicks <40 )   
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def motor_control_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<300,'RT'] = np.nan
    df_raw.loc[df_raw.RT>2250,'RT'] = np.nan

    medianRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    numCorrect = [None]* len(df.user_id)
    numSemiCorrect = [None]* len(df.user_id)
    numIncorr = [None]* len(df.user_id)
    numberNAs = [None]* len(df.user_id)
    meanDistance = [None]* len(df.user_id)
    meanCorrDistance = [None]* len(df.user_id)
    medianDistance = [None]* len(df.user_id)
    distanceScore = [None]* len(df.user_id)

    for count,id in enumerate(df.user_id):
        df_raw_temp = df_raw[df_raw.user_id == id]
        df_raw_temp.reset_index(drop=True,inplace=True)
        
        score_temp = [None] * len(df_raw_temp.distance)
        score_temp = pd.DataFrame(score_temp)
        score_temp.iloc[df_raw_temp.distance>75] = 0
        score_temp.iloc[(df_raw_temp.distance>25) & (df_raw_temp.distance<=75)] = 1
        score_temp.iloc[(df_raw_temp.distance>=0) & (df_raw_temp.distance<=25)] = 2
        
        numCorrect[count] = np.sum(score_temp.iloc[:,0] == 2) if (score_temp.iloc[:,0] == 2).any() else 0
        numSemiCorrect[count] = np.sum(score_temp.iloc[:,0] == 1) if (score_temp.iloc[:,0] == 1).any() else 0
        numIncorr[count] = np.sum(score_temp.iloc[:,0] == 0) if (score_temp.iloc[:,0] == 0).any() else 0
        
        meanDistance[count] = np.nanmean(df_raw_temp.distance)
        meanCorrDistance[count]= np.nanmean(df_raw_temp[score_temp.iloc[:,0] == 2].distance) if (score_temp.iloc[:,0] == 2).any() else np.nan
        medianDistance[count] = np.nanmedian(df_raw_temp.distance)
        
        meanRTs[count]= np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count]= np.nanmean(df_raw_temp[score_temp.iloc[:,0] < 2].RT) if (score_temp.iloc[:,0] < 2).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[score_temp.iloc[:,0] == 2].RT) if (score_temp.iloc[:,0] == 2).any() else np.nan

        medianRTs[count]= np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count]= np.nanmedian(df_raw_temp[score_temp.iloc[:,0] < 2].RT) if (score_temp.iloc[:,0] < 2).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[score_temp.iloc[:,0] == 2].RT) if (score_temp.iloc[:,0] == 2).any() else np.nan
        
        numberNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        distanceScore[count] = sum(score_temp)
        

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numCorrect":numCorrect, "numSemiCorrect":numSemiCorrect, "numIncorr":numIncorr, "numberNAs":numberNAs, "meanDistance":meanDistance, "meanCorrDistance":meanCorrDistance, "medianDistance":medianDistance})
    df["SummaryScore"] = df_raw_2.numCorrect
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    df["numCorrect"] = df_raw_2.numCorrect
    df["numSemiCorrect"] = df_raw_2.numSemiCorrect
    df["numIncorr"] = df_raw_2.numIncorr
    df["meanDistance"] = df_raw_2.meanDistance
    df["meanCorrDistance"] = df_raw_2.meanCorrDistance
    df["medianDistance"] = df_raw_2.medianDistance
    df["numberNAs"] = df_raw_2.numberNAs

    exc = (df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.numCorrect <= 15)   
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

def taskrecall_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<1000,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        errors[count] = sum(df_raw_temp.correct == False)
        scores[count] = sum(df_raw_temp.correct)
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)
        
    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})
    df["SummaryScore"] = df_raw_2.score
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    df["numberNAs"] = df_raw_2.numNAs

    exc = (df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore <2 )   
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.Level.replace(5, np.nan, inplace=True)
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

def auditory_attention_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<300,'RT'] = np.nan
    df_raw.loc[df_raw.RT>3000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    errorCome = [None] * len(df.user_id)
    errorBye = [None] * len(df.user_id)
    errorUp = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        scores[count] = sum(df_raw_temp.Correct)
        errors[count] = sum(df_raw_temp.Correct == False)
        errorCome[count] = (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['come'] + (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['go']
        errorBye[count] = (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['bye'] + (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['hi']
        errorUp[count] = (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['up'] + (df_raw_temp.groupby('Words').Correct.count() - df_raw_temp.groupby('Words').Correct.sum()).loc['down']
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.Correct == False].RT) if (~df_raw_temp.Correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.Correct == True].RT) if (df_raw_temp.Correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.Correct == False].RT) if (~df_raw_temp.Correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.Correct == True].RT) if (df_raw_temp.Correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "errorCome":errorCome, "errorBye":errorBye, "errorUp":errorUp})
    df["SummaryScore"] = df_raw_2.score
    df["CorrectSummary"] = df_raw_2.score
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["errorCome"] = df_raw_2.errorCome
    df["errorBye"] = df_raw_2.errorBye
    df["errorUp"] = df_raw_2.errorUp

    df["numberNAs"] = df_raw_2.numNAs

    exc = (df.timeOffScreen > 10000) | (df.focusLossCount > 2)  | (df.numberNAs < 0.3) | (df.numberNAs > 0.8) | (df.SummaryScore < 25) | (df.errorCome > 6 ) | (df.errorBye > 6) | (df.errorUp > 6 )
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw.Words.replace('undefined', np.nan, inplace=True)
    df_raw.dropna(subset=['Words'], inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

def calculation_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<600,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        scores[count] = sum(df_raw_temp.correct)
        errors[count] = sum(df_raw_temp.correct == False)

        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})
    df["SummaryScore"] = df_raw_2.score
    df["totalCorrect"] = df_raw_2.score
    df["errors"] = df_raw_2.errors

    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["numberNAs"] = df_raw_2.numNAs


    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore<4) | (df.numberNAs > 0.5) | (df.errors >=8) | (df.SummaryScore==9)) 
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def blocks_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<250,'RT'] = np.nan
    df_raw.loc[df_raw.RT>20000,'RT'] = np.nan

    trainingExists = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    errorDrop = [None] * len(df.user_id)
    maxConsecCorrect = [None] * len(df.user_id)
    maxConsecIncorrect = [None] * len(df.user_id)
    sumSkippedTrials = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        if (df_raw_temp['Num Crates'] == 4).any():
            trainingExists[count] = True
        else:
            trainingExists[count] = False
        
        if trainingExists[count]:
            
            df_raw_temp = df_raw_temp[df_raw_temp.Practice == False]
            
            sumSkippedTrials[count] = sum(df_raw_temp.NoResponse) if not df_raw_temp.empty else np.nan

            maxConsecCorrect[count] = max(df_raw_temp['ConsecutiveCorrect']) if not df_raw_temp.empty else np.nan
            maxConsecIncorrect[count] = max(df_raw_temp['ConsecutiveIncorrect']) if not df_raw_temp.empty else np.nan
            
            error_temp =  df_raw_temp['Num Drops'] > 0
            errorDrop[count] = (~((df_raw_temp[error_temp]).correct)).sum() if not df_raw_temp.empty else np.nan
        
            errors[count] = sum(df_raw_temp.correct == False) if not df_raw_temp.empty else np.nan
            meanRTs[count] = np.nanmean(df_raw_temp.RT)
            meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
            meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan           
            medianRTs[count] = np.nanmedian(df_raw_temp.RT)
            medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
            medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan

        else:
            
            maxConsecCorrect[count] = max(df_raw_temp['ConsecutiveCorrect'])
            maxConsecIncorrect[count] = max(df_raw_temp['ConsecutiveIncorrect'])
            
            sumSkippedTrials[count] = sum(df_raw_temp.NoResponse)
            
            error_temp =  df_raw_temp['Num Drops'] > 0
            errorDrop[count] = (~((df_raw_temp[error_temp]).correct)).sum()
            
            errors[count] = sum(df_raw_temp.correct == False)
            meanRTs[count] = np.nanmean(df_raw_temp.RT)
            meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
            meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan           
            medianRTs[count] = np.nanmedian(df_raw_temp.RT)
            medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
            medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan

        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "errorDrop":errorDrop, "maxConsecCorrect":maxConsecCorrect, "trainingExists":trainingExists, "maxConsecIncorrect":maxConsecIncorrect})
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    df["numberNAs"] = df_raw_2.numNAs
    df["errorsWithDrop"] = df_raw_2.errorDrop
    df["maxConsecCorrect"] = df_raw_2.maxConsecCorrect
    df["maxConsecIncorrect"] = df_raw_2.maxConsecIncorrect
    df["trainingExists"] = df_raw_2.trainingExists
    df["totalErrors"] = df_raw_2.errors

    for index,sub in df.iterrows():
        if sub.trainingExists:
            if sub.correctTotal < 3:
                df.loc[index,'correctTotal'] = 0
            else:
                df.loc[index,'correctTotal'] = sub.correctTotal - 3
    df['SummaryScore'] = df['correctTotal']

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore<3))
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def gesture_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<1500,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    sumSkippedTrials = [None] * len(df.user_id)
    sumAudioPlayed = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        scores[count] = sum(df_raw_temp.correct)
        errors[count] = sum(df_raw_temp.correct == False)
        
        sumSkippedTrials[count] = sum(df_raw_temp.NoResponse) if not df_raw_temp.empty else np.nan
        sumAudioPlayed[count] = df_raw_temp.loc[:,'playAgain1':'playAgain4'].sum(axis=1).sum()
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "sumSkippedTrials":sumSkippedTrials, "sumAudioPlayed":sumAudioPlayed})
    df["SummaryScore"] = df_raw_2.score
    df["totalCorrect"] = df_raw_2.score
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    df["numberNAs"] = df_raw_2.numNAs
    df["sumSkippedTrials"] = df_raw_2.sumSkippedTrials
    df["sumAudioPlayed"] = df_raw_2.sumAudioPlayed

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df_raw_2.sumSkippedTrials >=3) | (df.sumAudioPlayed>2) | (df.numberNAs >= 0.375) | (df.SummaryScore < 6)) 
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def rule_learning_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<400,'RT'] = np.nan
    df_raw.loc[df_raw.RT>10000,'RT'] = np.nan

    df.sort_values(by="user_id", inplace=True)
    score = [None]*len(df.user_id)
    id_temp = [None]*len(df.user_id)
    for count,sub in df.iterrows():
        id_temp[count] = sub.user_id
        sub = sub.responses.split(',')
        sub = [x.replace(' ','') for x in sub]
        sub = [x.replace('[','') for x in sub]
        sub = [x.replace(']','') for x in sub]
        sub = [int(x) for x in sub]
        sub = [trial == 0 for trial in sub]
        if any(sub):
            score[count] = [sub.index(True)][0]
        else:
            score[count] = len(sub)
            
    df_temp = pd.DataFrame({"user_id":id_temp, "MaxLevelReached":score})
    df = pd.merge(df, df_temp, on='user_id')
    df['SummaryScore'] = df.MaxLevelReached

    scores = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        scores[count] = len(df_raw_temp.correct)
        
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})
    df["totalResponses"] = df_raw_2.score
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["numberNAs"] = df_raw_2.numNAs

    df.loc[df['completed']==1,'SummaryScore'] = 10

    errorsConditions = [None]*len(df.user_id)
    errorsTrials = [None]*len(df.user_id)
    id_temp = [None]*len(df.user_id)
    for count,sub in df.iterrows():
        id_temp[count] = sub.user_id
        sub = sub.responses.split(',')
        sub = [x.replace(' ','') for x in sub]
        sub = [x.replace('[','') for x in sub]
        sub = [x.replace(']','') for x in sub]
        sub = [int(x) for x in sub]
        sub_temp1 = [trial > 7 for trial in sub]
        if any(sub_temp1):
            errorsConditions[count] = sum(sub_temp1)
            errorsTrials[count] = sum([trial - 7 for trial in sub if trial > 7])
        else:
            errorsConditions[count] = 0
            errorsTrials[count] = 0
        
    df_temp = pd.DataFrame({"user_id":id_temp, "conditionsWithShiftErrors":errorsConditions, "trialsWithShiftErrors":errorsTrials})
    df = pd.merge(df, df_temp, on='user_id')

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.SummaryScore < 3) | (df.trialsWithShiftErrors >= 30)) 
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    
    return df,df_raw  


  
def oddoneout_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<700,'RT'] = np.nan
    df_raw.loc[df_raw.RT>20000,'RT'] = np.nan

    df_raw.correct.replace(np.nan, False, inplace=True)
    for index, trial in df_raw.iterrows():
        if trial.NoResponse == True:
            df_raw.loc[index, 'correct'] = False
    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    ids_temp = [None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)
    sumSkippedTrials = [None] * len(df.user_id)
    maxConsecCorrect = [None] * len(df.user_id)
    maxConsecIncorrect = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw.loc[df_raw.user_id == id, :]
        if not df_raw_temp.empty:
            startTask = df_raw_temp["Time Resp Enabled"].iloc[0]
            endTask = df_raw_temp["Time Resp Enabled"].iloc[len(df_raw_temp.RT)-1]
            reasonableLength = 1000 * 60 * 60 * 3 # 3 hours
            if (endTask - startTask) > reasonableLength:
                #print(f'Trials were removed from {id}' )
                oddTrials = (df_raw_temp["Time Resp Enabled"] - startTask) > reasonableLength
                oddTrials = df_raw_temp[oddTrials].index
                df_raw = df_raw.drop(oddTrials)
                df_raw_temp = df_raw_temp.drop(oddTrials)
                
        ids_temp[count] = id
        
        scores[count] = sum(df_raw_temp.correct)
        errors[count] = sum(df_raw_temp.correct == False)
                
        sumSkippedTrials[count] = sum(df_raw_temp.NoResponse) if not df_raw_temp.empty else np.nan

        maxConsecCorrect[count] = max(df_raw_temp['ConsecutiveCorrect']) if not df_raw_temp.empty else np.nan
        maxConsecIncorrect[count] = max(df_raw_temp['ConsecutiveIncorrect']) if not df_raw_temp.empty else np.nan
            
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "sumSkippedTrials":sumSkippedTrials, "maxConsecCorrect":maxConsecCorrect, "maxConsecIncorrect":maxConsecIncorrect})
    df["SummaryScore"] = df_raw_2.score
    df["ncorrect"] = df_raw_2.score
    df["nincorrect"] = df_raw_2.errors

    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrCRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["numberNAs"] = df_raw_2.numNAs
    df["sumSkippedTrials"] = df_raw_2.sumSkippedTrials
    df["maxConsecCorrect"] = df_raw_2.maxConsecCorrect
    df["maxConsecIncorrect"] = df_raw_2.maxConsecIncorrect

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.ncorrect < 6) | (df.nincorrect >= 10) | (df.maxConsecIncorrect >=6) | (df.numberNAs >= 0.375) ) 
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def comprehension_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<650,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    level1_errors = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    sumSkippedTrials = [None] * len(df.user_id)
    sumAudioPlayed = [None] * len(df.user_id)
    maxConsecCorrect = [None] * len(df.user_id)
    maxConsecIncorrect = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        
        scores[count] = sum(df_raw_temp.correct)
        errors[count] = sum(df_raw_temp.correct == False)
        level1_errors[count]= sum(df_raw_temp[df_raw_temp.Level == 1].correct == False)
        
        sumSkippedTrials[count] = sum(df_raw_temp['No response']) if not df_raw_temp.empty else np.nan
        sumAudioPlayed[count] = sum(df_raw_temp.PlayedAgain) if not df_raw_temp.empty else np.nan

        maxConsecCorrect[count] = max(df_raw_temp['ConsecutiveCorrect']) if not df_raw_temp.empty else np.nan
        maxConsecIncorrect[count] = max(df_raw_temp['ConsecutiveIncorrect']) if not df_raw_temp.empty else np.nan

        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "sumSkippedTrials":sumSkippedTrials, "sumAudioPlayed":sumAudioPlayed, "maxConsecCorrect":maxConsecCorrect, "maxConsecIncorrect":maxConsecIncorrect, "level1_errors":level1_errors})
    df["SummaryScore"] = df_raw_2.score
    df["totalCorrect"] = df_raw_2.score
    df['totalErrors'] = df_raw_2.errors
    df['level1Errors'] = df_raw_2.level1_errors

    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT

    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT

    df["numberNAs"] = df_raw_2.numNAs
    df["sumSkippedTrials"] = df_raw_2.sumSkippedTrials
    df["maxConsecCorrect"] = df_raw_2.maxConsecCorrect
    df["maxConsecIncorrect"] = df_raw_2.maxConsecIncorrect
    df["sumAudioPlayed"] = df_raw_2.sumAudioPlayed

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2)| (df.numberNAs > 0.5) | (df.SummaryScore <= 12) | (df.level1Errors > 1))
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def trailmaking_preproc(df,df2,df3,df_raw,df_raw2,df_raw3, task_name):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>10000,'RT'] = np.nan
    
    df_raw2.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw2.loc[df_raw.RT>10000,'RT'] = np.nan

    df_raw3.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw3.loc[df_raw.RT>10000,'RT'] = np.nan


    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
    errorsPractice = [None] * len(df.user_id)
    maxConsecIncorrectPractice = [None] * len(df.user_id)
    sumSkippedTrials = [None] * len(df.user_id)
    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
    meanErrorRTs =[None] * len(df.user_id)
    meanCorrRTs =[None] * len(df.user_id)
    medianErrorRTs =[None] * len(df.user_id)
    medianCorrRTs =[None] * len(df.user_id)
    numNAs = [None] * len(df.user_id)

    for count,id in enumerate(df.user_id):
        
        df_raw_temp = df_raw[df_raw.user_id == id]
        df_raw_temp = df_raw_temp[df_raw_temp.ImageNumberMainTask != 'undefined']
        
        scores[count] = sum(df_raw_temp.correct)
        errors[count] = sum(df_raw_temp.correct == False)
        
        sumSkippedTrials[count] = sum(df_raw_temp['NoResponse']) if not df_raw_temp.empty else np.nan

        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw_temp.RT.isna().sum() / len(df_raw_temp.RT)

    df_raw_2 = pd.DataFrame({"user_id":df.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "sumSkippedTrials":sumSkippedTrials})
    df["SummaryScore"] = df_raw_2.errors
    df["totalCorrect"] = df_raw_2.score
    df['totalIncorrect'] = df_raw_2.errors
    df["medianRT"] = df_raw_2.medianRT
    df["medianCorrRT"] = df_raw_2.medianCorrRT
    df["medianErrorRT"] = df_raw_2.medianErrorRT
    df["meanRT"] = df_raw_2.meanRT
    df["meanCorrRT"] = df_raw_2.meanCorrRT
    df["meanErrorRT"] = df_raw_2.meanErrorRT
    df["numberNAs"] = df_raw_2.numNAs

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.totalIncorrect > 10) )
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)


    scores = [None] * len(df2.user_id)
    errors = [None] * len(df2.user_id)
    errorsPractice = [None] * len(df2.user_id)
    maxConsecIncorrectPractice = [None] * len(df2.user_id)
    sumSkippedTrials = [None] * len(df2.user_id)
    meanRTs =[None] * len(df2.user_id)
    medianRTs =[None] * len(df2.user_id)
    meanErrorRTs =[None] * len(df2.user_id)
    meanCorrRTs =[None] * len(df2.user_id)
    medianErrorRTs =[None] * len(df2.user_id)
    medianCorrRTs =[None] * len(df2.user_id)
    numNAs = [None] * len(df2.user_id)

    for count,id in enumerate(df2.user_id):
        
        df_raw2_temp = df_raw2[df_raw2.user_id == id]
        df_raw2_temp = df_raw2_temp[df_raw2_temp.ImageNumberMainTask != 'undefined']
        
        scores[count] = sum(df_raw2_temp.correct)
        errors[count] = sum(df_raw2_temp.correct == False)
        
        sumSkippedTrials[count] = sum(df_raw2_temp['NoResponse']) if not df_raw2_temp.empty else np.nan

        meanRTs[count] = np.nanmean(df_raw2_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw2_temp[df_raw2_temp.correct == False].RT) if (~df_raw2_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw2_temp[df_raw2_temp.correct == True].RT) if (df_raw2_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw2_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw2_temp[df_raw2_temp.correct == False].RT) if (~df_raw2_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw2_temp[df_raw2_temp.correct == True].RT) if (df_raw2_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw2_temp.RT.isna().sum() / len(df_raw2_temp.RT)

    df_raw2_2 = pd.DataFrame({"user_id":df2.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs, "sumSkippedTrials":sumSkippedTrials})
    df2["SummaryScore"] = df_raw2_2.errors
    df2["totalCorrect"] = df_raw2_2.score
    df2['totalIncorrect'] = df_raw2_2.errors
    df2["medianRT"] = df_raw2_2.medianRT
    df2["medianCorrRT"] = df_raw2_2.medianCorrRT
    df2["medianErrorRT"] = df_raw2_2.medianErrorRT
    df2["meanRT"] = df_raw2_2.meanRT
    df2["meanCorrRT"] = df_raw2_2.meanCorrRT
    df2["meanErrorRT"] = df_raw2_2.meanErrorRT
    df2["numberNAs"] = df_raw2_2.numNAs

    exc = ((df2.timeOffScreen > 10000) | (df2.focusLossCount > 2) | (df2.totalIncorrect > 10) )
    df2.drop(df2[exc].index, inplace=True)
    df2.reset_index(drop=True,inplace=True)
    df_raw2 = df_raw2[df_raw2.user_id.isin(df2.user_id)]
    df_raw2.reset_index(drop=True,inplace=True)


    scores = [None] * len(df3.user_id)
    errors = [None] * len(df3.user_id)
    sumSkippedTrials = [None] * len(df3.user_id)
    meanRTs =[None] * len(df3.user_id)
    medianRTs =[None] * len(df3.user_id)
    meanErrorRTs =[None] * len(df3.user_id)
    meanCorrRTs =[None] * len(df3.user_id)
    medianErrorRTs =[None] * len(df3.user_id)
    medianCorrRTs =[None] * len(df3.user_id)
    numNAs = [None] * len(df3.user_id)

    for count,id in enumerate(df3.user_id):
        
        df_raw3_temp = df_raw3[df_raw3.user_id == id]
        
        scores[count] = sum(df_raw3_temp.correct)
        errors[count] = sum(df_raw3_temp.correct == False)
        
        sumSkippedTrials[count] = sum(df_raw3_temp['NoResponse']) if not df_raw3_temp.empty else np.nan

        meanRTs[count] = np.nanmean(df_raw3_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw3_temp[df_raw3_temp.correct == False].RT) if (~df_raw3_temp.correct).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw3_temp[df_raw3_temp.correct == True].RT) if (df_raw3_temp.correct).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw3_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw3_temp[df_raw3_temp.correct == False].RT) if (~df_raw3_temp.correct).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw3_temp[df_raw3_temp.correct == True].RT) if (df_raw3_temp.correct).any() else np.nan
        
        numNAs[count] = df_raw3_temp.RT.isna().sum() / len(df_raw3_temp.RT)

    df_raw3_2 = pd.DataFrame({"user_id":df3.user_id, "score":scores, "errors":errors, "meanRT":meanRTs, "medianRT":medianRTs, "medianErrorRT":medianErrorRTs, "medianCorrRT":medianCorrRTs, "meanErrorRT":meanErrorRTs, "meanCorrRT":meanCorrRTs, "numNAs":numNAs})
    df3["SummaryScore"] = df_raw3_2.errors
    df3["totalCorrect"] = df_raw3_2.score
    df3['totalIncorrect'] = df_raw3_2.errors

    df3["medianRT"] = df_raw3_2.medianRT
    df3["medianCorrRT"] = df_raw3_2.medianCorrRT
    df3["medianErrorRT"] = df_raw3_2.medianErrorRT

    df3["meanRT"] = df_raw3_2.meanRT
    df3["meanCorrRT"] = df_raw3_2.meanCorrRT
    df3["meanErrorRT"] = df_raw3_2.meanErrorRT

    df3["numberNAs"] = df_raw3_2.numNAs

    exc = ((df3.timeOffScreen > 10000) | (df3.focusLossCount > 2) | (df3.totalIncorrect > 10) )
    df3.drop(df3[exc].index, inplace=True)
    df3.reset_index(drop=True,inplace=True)
    df_raw3 = df_raw3[df_raw3.user_id.isin(df3.user_id)]
    df_raw3.reset_index(drop=True,inplace=True)


    df = df.loc[:, ['user_id', 'totalCorrect', 'totalIncorrect','meanCorrRT','medianCorrRT']]
    df2 = df2.loc[:, ['user_id', 'totalCorrect', 'totalIncorrect','meanCorrRT','medianCorrRT']]
    df3.rename(columns={"totalCorrect": "totalCorrect_level3", "totalIncorrect": "totalIncorrect_level3",'meanCorrRT':'meanCorrRT_level3','medianCorrRT':'medianCorrRT_level3'}, inplace=True)
    df.rename(columns={"totalCorrect": "totalCorrect_level1", "totalIncorrect": "totalIncorrect_level1",'meanCorrRT':'meanCorrRT_level1','medianCorrRT':'medianCorrRT_level1'}, inplace=True)
    df2.rename(columns={"totalCorrect": "totalCorrect_level2", "totalIncorrect": "totalIncorrect_level2",'meanCorrRT':'meanCorrRT_level2','medianCorrRT':'medianCorrRT_level2'}, inplace=True)
    df_temp = df.merge(df2, on='user_id', how='outer').merge(df3, on='user_id', how='outer')
    df_temp.totalIncorrect_level1.fillna(0, inplace=True)
    df_temp.totalIncorrect_level2.fillna(0, inplace=True)
    df_temp.totalIncorrect_level3.fillna(6, inplace=True)
    df_temp['switchCostAccuracy'] = df_temp.totalCorrect_level3 - df_temp.totalCorrect_level2 - df_temp.totalCorrect_level1
    df_temp['switchCostErrors'] = df_temp.totalIncorrect_level3 - (df_temp.totalIncorrect_level2 + df_temp.totalIncorrect_level1)/2 
    df_temp['switchCostMedianCorrRT'] = df_temp.medianCorrRT_level3 - (df_temp.medianCorrRT_level2 + df_temp.medianCorrRT_level1)/2
    df_temp['switchCostMeanCorrRT'] = df_temp.meanCorrRT_level3 - (df_temp.meanCorrRT_level2 + df_temp.meanCorrRT_level1)/2
    df_temp.loc[df_temp['switchCostErrors'] <0, 'switchCostErrors'] = 0
    df_temp.loc[df_temp['switchCostAccuracy'] <0, 'switchCostAccuracy'] = 0
    df_temp.loc[df_temp['switchCostMedianCorrRT'] <0, 'switchCostMedianCorrRT'] = 0
    df_temp.loc[df_temp['switchCostMeanCorrRT'] <0, 'switchCostMeanCorrRT'] = 0

    df_temp_raw = pd.concat([df_raw,df_raw2,df_raw3],ignore_index=True,axis=0)   
    df_temp_raw.reset_index(drop=True,inplace=True)

    exc = (df_temp.switchCostErrors >6) 
    df_temp.drop(df_temp[exc].index, inplace=True)
    df_temp.reset_index(drop=True,inplace=True)
    df_temp['taskID'] = task_name
    df_temp_raw['taskID'] = task_name
    
    return df_temp,df_temp_raw  