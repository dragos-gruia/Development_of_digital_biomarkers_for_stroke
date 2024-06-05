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

def main_preprocessing(root_path, list_of_tasks, list_of_questionnaires, list_of_speech, clinical_information=None, patient_data_folders=['/data_patients_v1','/data_patients_v2'],
                       folder_structure=['/summary_data','/trial_data','/speech'], output_clean_folder ='/data_patients_cleaned', merged_data_folder ='/data_patients_merged', clean_file_extension='_cleaned', data_format='.csv' ):

    print('Starting preprocessing...')
    
    os.chdir(root_path)

    df_demographics = pd.DataFrame()
    df_iadl = pd.DataFrame()
    df_combined = pd.DataFrame()
    
    print('Merging data across sites...', end="", flush=True)
    
    merge_patient_data_across_sites(root_path, folder_structure, patient_data_folders, list_of_tasks,list_of_questionnaires,list_of_speech, data_format, merged_data_folder)
    
    print('Done')
    
    if list_of_tasks != None:
        for task_name in list_of_tasks:

            print(f'Pre-processing {task_name}...', end="", flush=True)
            df,df_raw = remove_general_outliers(root_path, merged_data_folder, task_name, data_format, folder_structure)

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
                    df2,df_raw2 = remove_general_outliers(root_path, merged_data_folder, (f'{task_name}2'),  data_format, folder_structure)
                    df3,df_raw3 = remove_general_outliers(root_path, merged_data_folder, (f'{task_name}3'), data_format, folder_structure)
                    df,df_raw = trailmaking_preproc(df,df2,df3,df_raw,df_raw2,df_raw3,task_name)
                    
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
            
            output_preprocessed_data(df,df_raw, root_path, output_clean_folder, folder_structure, clean_file_extension, data_format)
            print('Done')
    else:
        print('No tasks were provided.') 
        
                   
    if list_of_questionnaires != None:
        for questionnaire_name in list_of_questionnaires:

            print(f'Pre-processing {questionnaire_name}...', end="", flush=True)
            
            match questionnaire_name:
                
                case 'q_IC3_demographics':
                    df_demographics = demographics_preproc(root_path, merged_data_folder, questionnaire_name, folder_structure, data_format, clean_file_extension, output_clean_folder)                   
                    print('Done')
                    
                case 'q_IC3_IADL':
                    df_iadl = iadl_preproc(root_path, merged_data_folder,questionnaire_name,folder_structure,data_format,clean_file_extension, output_clean_folder)
                    
                case _:
                    print(f'Questionnaire {questionnaire_name} does not have a specific preprocessing function.')
                    

    else:
            print('No questionnaires were provided.')            
    
    if not df_demographics.empty:  
        df_combined = combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure, list_of_tasks, df_demographics, clean_file_extension, data_format, clinical_information)
    
    if not df_iadl.empty:     
        df_combined = df_combined.merge(df_iadl, on='user_id', how='left')
    
    print('Preprocessing complete.')   
    
    df_combined.to_excel(f'summary_cognition_and_demographics.xlsx')
        
    return df_combined   




def merge_patient_data_across_sites(root_path, folder_structure, patient_data_folders, list_of_tasks,list_of_questionnaires,list_of_speech, data_format, merged_data_folder):
        
    os.chdir(root_path)

    # Create folder structure
    
    if os.path.isdir(merged_data_folder[1:]) == False:
        os.mkdir(merged_data_folder[1:])
    os.chdir(merged_data_folder[1:])

    if os.path.isdir(folder_structure[0][1:]) == False:
        os.mkdir(folder_structure[0][1:])
        
    if os.path.isdir(folder_structure[1][1:]) == False:
        os.mkdir(folder_structure[1][1:])
    
    if os.path.isdir(folder_structure[2][1:]) == False:
        os.mkdir(folder_structure[2][1:])
        

    # Merge data from clinical tests
    
    if list_of_tasks != None:
        for taskName in list_of_tasks:
                
            os.chdir("..")
            os.chdir(patient_data_folders[0][1:])
            
            df_v1 = pd.read_csv(f'.{folder_structure[0]}/{taskName}{data_format}', low_memory=False)
            df_v1_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw{data_format}'), low_memory=False)
            
            if taskName == "IC3_NVtrailMaking": 
                
                df2_v1 = pd.read_csv(f'.{folder_structure[0]}/{taskName}2{data_format}', low_memory=False)
                df2_v1_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}2_raw{data_format}'), low_memory=False)
                
                df3_v1 = pd.read_csv(f'.{folder_structure[0]}/{taskName}3{data_format}', low_memory=False)
                df3_v1_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}3_raw{data_format}'), low_memory=False)
            
            os.chdir("..")
            os.chdir(patient_data_folders[1][1:])
            
            # Special case for IDED task that has two versions
            
            if taskName == "IC3_i4i_IDED": 
                df_v2 = pd.read_csv((f'.{folder_structure[0]}/{taskName}2.csv'), low_memory=False)
                df_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}2_raw.csv'), low_memory=False)
            else:   
                df_v2 = pd.read_csv(f'.{folder_structure[0]}/{taskName}{data_format}', low_memory=False)
                df_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw.csv'), low_memory=False)
            
            if taskName == "IC3_NVtrailMaking": 
                
                df2_v2 = pd.read_csv(f'.{folder_structure[0]}/{taskName}2{data_format}', low_memory=False)
                df2_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}2_raw{data_format}'), low_memory=False)
                
                df3_v2 = pd.read_csv(f'.{folder_structure[0]}/{taskName}3{data_format}', low_memory=False)
                df3_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}3_raw{data_format}'), low_memory=False)      
                
                df2 = pd.concat([df2_v1, df2_v2], ignore_index=True)
                df2_raw = pd.concat([df2_v1_raw, df2_v2_raw], ignore_index=True) 
                
                df3 = pd.concat([df3_v1, df3_v2], ignore_index=True)
                df3_raw = pd.concat([df3_v1_raw, df3_v2_raw], ignore_index=True) 
                     
            
            df = pd.concat([df_v1, df_v2], ignore_index=True)
            df_raw = pd.concat([df_v1_raw, df_v2_raw], ignore_index=True)
            
            os.chdir("..")
            os.chdir(merged_data_folder[1:])
            
            if taskName == "IC3_NVtrailMaking":
                df2.to_csv(f'.{folder_structure[0]}/{taskName}2{data_format}', index=False)
                df2_raw.to_csv(f'.{folder_structure[1]}/{taskName}2_raw.csv', index=False)                

                df3.to_csv(f'.{folder_structure[0]}/{taskName}3{data_format}', index=False)
                df3_raw.to_csv(f'.{folder_structure[1]}/{taskName}3_raw.csv', index=False) 
                
            df.to_csv(f'.{folder_structure[0]}/{taskName}{data_format}', index=False)
            df_raw.to_csv(f'.{folder_structure[1]}/{taskName}_raw.csv', index=False)
            print(f'Merged {taskName}')
        
    # Merge data from speech
    
    if list_of_speech != None:    
        for taskName in list_of_speech:
            
            os.chdir("..")
            os.chdir(patient_data_folders[0][1:])
            df_v1_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw.csv'), low_memory=False)
            
            os.chdir("..")
            os.chdir(patient_data_folders[1][1:])
            df_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw.csv'), low_memory=False)

            df_raw = pd.concat([df_v1_raw, df_v2_raw], ignore_index=True)
            
            os.chdir("..")
            os.chdir(merged_data_folder[1:])
            df_raw.to_csv(f'.{folder_structure[1]}/{taskName}_raw.csv', index=False)
            print(f'Merged {taskName}')
        
    # Merge data from questionnaires
    
    if list_of_questionnaires != None:
        for taskName in list_of_questionnaires:
                
            os.chdir("..")
            os.chdir(patient_data_folders[0][1:])
            df_v1 = pd.read_csv(f'.{folder_structure[0]}/{taskName}_questionnaire.csv', low_memory=False)
            df_v1_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw.csv'), low_memory=False)
            
            os.chdir("..")
            os.chdir(patient_data_folders[1][1:])
            df_v2 = pd.read_csv(f'.{folder_structure[0]}/{taskName}_questionnaire.csv', low_memory=False)
            df_v2_raw = pd.read_csv((f'.{folder_structure[1]}/{taskName}_raw.csv'), low_memory=False)
            
            df = pd.concat([df_v1, df_v2], ignore_index=True)
            df_raw = pd.concat([df_v1_raw, df_v2_raw], ignore_index=True)
                
            os.chdir("..")
            os.chdir(merged_data_folder[1:])
            df.to_csv(f'.{folder_structure[0]}/{taskName}{data_format}', index=False)
            df_raw.to_csv(f'.{folder_structure[1]}/{taskName}_raw.csv', index=False)
            print(f'Merged {taskName}')
                
        return None


def combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure, list_of_tasks, df_demographics, clean_file_extension, data_format, clinical_information):
    
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
              
        if file == 'IC3_Orientation':    
            temp_healthy_cog = temp_healthy_cog.loc[:,['user_id','startTime','SummaryScore']]
        else:
            temp_healthy_cog = temp_healthy_cog.loc[:,['user_id','SummaryScore']]
            
        temp_healthy_cog.rename(columns={'SummaryScore':task_id}, inplace=True)
        
        df_demographics= pd.merge(df_demographics, temp_healthy_cog,  on="user_id", how="left")


    # Link patients based on id and timepoint
    
    df_demographics['ID'] = np.nan
    df_demographics['timepoint'] = np.nan
    df_demographics.user_id = df_demographics.user_id.astype(str) 
    df_demographics.index = df_demographics.user_id
    
    for index,ids in enumerate(df_demographics.user_id):
        if 'session1' in ids:
            df_demographics.loc[ids, 'timepoint'] = 1 
        elif 'session2' in ids:
            df_demographics.loc[ids, 'timepoint'] = 2
        elif 'session3' in ids:
            df_demographics.loc[ids, 'timepoint'] = 3
        elif 'session4' in ids:
            df_demographics.loc[ids, 'timepoint'] = 4
            
        if len(ids.split('-')) == 3:
            
            temp_id = ids.split('-')[0]
            temp_id
            if 'ic3study' in temp_id:
                df_demographics.loc[ids, 'ID'] = temp_id[8:]
                #print(temp_id[8:])
            else:
                df_demographics.loc[ids, 'ID'] = temp_id
                #print(temp_id)
                
    
    df_demographics = fix_naming_errors(df_demographics)
    df_demographics = remove_technical_errors(df_demographics)
    
    # Add clinical information if available
    
    if clinical_information != None:
        df_clinical = pd.read_excel(clinical_information, sheet_name='Patient ') 
        df_clinical = df_clinical.loc[:,['STUDY ID','CVA date','CVA aetiology','vascular teritory','lesion site','Aphasia','CVA non-cognitive deficit', 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis']]
        df_clinical.rename(columns={'STUDY ID':'ID'}, inplace=True)
        df_clinical['ID'] = df_clinical['ID'].apply(lambda x: x if (x != x) else x.strip() ) #remove white spaces if not NaN
        df_combined = df_demographics.merge(df_clinical, on='ID', how='left')
        df_combined.drop_duplicates(subset='user_id', inplace=True)
        df_combined.dropna(subset=['startTime'], inplace=True)
        df_combined['date_of_ic3'] = df_combined['startTime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
        df_combined['date_of_ic3'] = pd.to_datetime(df_combined['date_of_ic3'])
        df_combined['CVA date'] = pd.to_datetime(df_combined['CVA date'])
        df_combined['days_since_stroke'] = (df_combined['date_of_ic3'] - df_combined['CVA date']).dt.days
        df_combined.to_excel('summary_cognition_and_demographics.xlsx')
        return df_combined
    else:
        df_demographics.to_excel('summary_cognition_and_demographics.xlsx')
        return df_demographics


def remove_technical_errors(df_demographics):
    
    df_demographics.drop(index='ic3study10039-session1-versionA',inplace=True)
    df_demographics.drop(index='ic3study00097-session1-versionA',inplace=True)
    df_demographics.loc[df_demographics.ID == 'Anon-4401C1A6EEB843FBA061A62EC8C1E47D', 'IC3_PearCancellation'] = np.nan
    df_demographics.loc[df_demographics.ID == 'Anon-4401C1A6EEB843FBA061A62EC8C1E47D', 'IC3_SemanticJudgment'] = np.nan
    df_demographics.loc[df_demographics.ID == '00008-session1-versionA', 'IC3_GestureRecognition'] = np.nan
    df_demographics.loc[df_demographics.ID == '00009-session1-versionA', 'IC3_SemanticJudgment'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00018-session1-versionA', 'IC3_rs_CRT'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00033-session1-versionA', 'IC3_rs_CRT'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00041-session1-versionA', 'IC3_GestureRecognition'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00090-session1-versionA', 'IC3_rs_CRT'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00095-session1-versionA', 'IC3_Comprehension'] = np.nan
    df_demographics.loc[df_demographics.ID == 'ic3study00124-session1-versionA', 'IC3_calculation'] = np.nan
    df_demographics.reset_index(drop=True, inplace=True)
    
    return df_demographics
        
def fix_naming_errors(df_demographics):
    
    df_demographics.loc['ic300005s1o1', 'ID'] = '00005'
    df_demographics.loc['ic300005s1o1', 'timepoint'] = 1
    df_demographics.loc['00008-session1-versionA', 'ID'] = '00008'
    df_demographics.loc['00008-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['Anon-4401C1A6EEB843FBA061A62EC8C1E47D', 'ID'] = '00007'
    df_demographics.loc['Anon-4401C1A6EEB843FBA061A62EC8C1E47D', 'timepoint'] = 1
    df_demographics.loc['00011-session1-versiona', 'ID'] = '00011'
    df_demographics.loc['00011-session1-versiona', 'timepoint'] = 1
    df_demographics.loc['00012-session1-versionA', 'ID'] = '00012'
    df_demographics.loc['00012-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['00009', 'ID'] = '00004'
    df_demographics.loc['00009', 'timepoint'] = 1
    df_demographics.loc['ic3study00015-session1-versionA', 'ID'] = '00015'
    df_demographics.loc['ic3study00015-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00016-session1-versionA', 'ID'] = '00016'
    df_demographics.loc['ic3study00016-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['bbrandon@hotmail.co.uk', 'ID'] = '00014'
    df_demographics.loc['bbrandon@hotmail.co.uk', 'timepoint'] = 1
    df_demographics.loc['simoko.hart@gmail.com', 'ID'] = '00017'
    df_demographics.loc['simoko.hart@gmail.com', 'timepoint'] = 1
    df_demographics.loc['ic3study00018-session1-versionA', 'ID'] = '00018'
    df_demographics.loc['ic3study00018-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00019-session1-versionA', 'ID'] = '00019'
    df_demographics.loc['ic3study00019-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00020-session1-versionA', 'ID'] = '00020'
    df_demographics.loc['ic3study00020-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00021-session1-versionA', 'ID'] = '00021'
    df_demographics.loc['ic3study00021-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00024-session1-versionA', 'ID'] = '00024'
    df_demographics.loc['ic3study00024-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00022-session1-versionA', 'ID'] = '00022'
    df_demographics.loc['ic3study00022-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00023-session1-versionA', 'ID'] = '00023'
    df_demographics.loc['ic3study00023-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00027-session1-versionA', 'ID'] = '00027'
    df_demographics.loc['ic3study00027-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00032-session1-versionA', 'ID'] = '00032'
    df_demographics.loc['ic3study00032-session1-versionA', 'timepoint'] = 2
    df_demographics.loc['00032-session1-versionA', 'ID'] = '00032'
    df_demographics.loc['00032-session1-versionA', 'timepoint'] = 3
    df_demographics.loc['ic3study00033-session1-versionA', 'ID'] = '00033'
    df_demographics.loc['ic3study00033-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00036-session1-versionA', 'ID'] = '00036'
    df_demographics.loc['ic3study00036-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00040-session1-versionA', 'ID'] = '00040'
    df_demographics.loc['ic3study00040-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00041-session1-versionA', 'ID'] = '00041'
    df_demographics.loc['ic3study00041-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00010-session2-versionA', 'ID'] = '00010'
    df_demographics.loc['ic3study00010-session2-versionA', 'timepoint'] = 2
    df_demographics.loc['ic3study00020-session2-versionB', 'ID'] = '00020'
    df_demographics.loc['ic3study00020-session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00042-session1-versionA', 'ID'] = '00042'
    df_demographics.loc['ic3study00042-session1-versionA', 'timepoint'] = 1
    df_demographics.loc['ic3study00031-session2-versionB', 'ID'] = '00031'
    df_demographics.loc['ic3study00031-session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00003-session2-versionB', 'ID'] = '00003'
    df_demographics.loc['ic3study00003-session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study20010session2-versionB', 'ID'] = '20010'
    df_demographics.loc['ic3study20010session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00098session2-versionB', 'ID'] = '00098'
    df_demographics.loc['ic3study00098session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00099session2-versionB', 'ID'] = '00099'
    df_demographics.loc['ic3study00099session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00070session3-versionA', 'ID'] = '00070'
    df_demographics.loc['ic3study00070session3-versionA', 'timepoint'] = 3
    df_demographics.loc['ic3study0110-session2-versionB', 'ID'] = '00110'
    df_demographics.loc['ic3study0110-session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study00014-session1-versionA', 'ID'] = '00014'
    df_demographics.loc['ic3study00014-session1-versionA', 'timepoint'] = 2
    df_demographics.loc['ic3study00004-session2-versionA', 'ID'] = '00004'
    df_demographics.loc['ic3study00004-session2-versionA', 'timepoint'] = 2
    df_demographics.loc['ic3study00003-session3', 'ID'] = '00003'
    df_demographics.loc['ic3study00003-session3', 'timepoint'] = 3
    df_demographics.loc['ic3study00096session2-versionB', 'ID'] = '00096'
    df_demographics.loc['ic3study00096session2-versionB', 'timepoint'] = 2
    df_demographics.loc['ic3study000865-session1-versionA', 'ID'] = '00086'
    df_demographics.loc['Anon-3E31422CD53341458C2EAA17599A1BEF', 'ID'] = '00064'
    df_demographics.loc['Anon-3E31422CD53341458C2EAA17599A1BEF', 'timepoint'] = 1
    df_demographics.loc['Anon-0CFAD094D29E484AA02DCE99FACECBC3', 'ID'] = '00097'
    df_demographics.loc['Anon-0CFAD094D29E484AA02DCE99FACECBC3', 'timepoint'] = 1
    df_demographics.loc['Anon-E3D074ECDB7E4856B7968501981E76BC', 'ID'] = '00079'
    df_demographics.loc['Anon-E3D074ECDB7E4856B7968501981E76BC', 'timepoint'] = 2
    df_demographics.loc['Anon-CAB8613480C04EBB8BE3BFFBE310558F', 'ID'] = '00042'
    df_demographics.loc['Anon-CAB8613480C04EBB8BE3BFFBE310558F', 'timepoint'] = 4
    df_demographics.loc['Anon-1BD5269BC73644E3907617937B23B7C1', 'ID'] = '00043'
    df_demographics.loc['Anon-1BD5269BC73644E3907617937B23B7C1', 'timepoint'] = 4
    df_demographics.loc['Anon-F98995DC24CD48EAAA0945FCCE70FA86', 'ID'] = '00018'
    df_demographics.loc['Anon-F98995DC24CD48EAAA0945FCCE70FA86', 'timepoint'] = 4
    
    #df_demographics.dropna(subset=['ID','timepoint','user_id'], inplace=True)
    df_demographics.dropna(subset=['user_id'], inplace=True)
    
    return df_demographics

        
def iadl_preproc(root_path, merged_data_folder,questionnaire_name,folder_structure,data_format,clean_file_extension, output_clean_folder):
    
    os.chdir(root_path + merged_data_folder)
    try:
        df_iadl = pd.read_csv(f'.{folder_structure[1]}/{questionnaire_name}_raw{data_format}', low_memory=False)
        
    except:
        print(f'Error in loading {questionnaire_name}. File might not exist.')
        return None        

    df_iadl.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_iadl.drop_duplicates(subset=['question','user_id'], keep='last', inplace=True)
    df_iadl.dropna(subset=['response'], inplace=True)
    df_iadl.reset_index(drop=True, inplace=True)
    
    df_iadl['score'] = df_iadl['response'].apply(lambda x: int(x[0]))
    
    df_iadl.loc[df_iadl.question == '<center>Abilitytousetelephone</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Abilitytousetelephone</center>', 'score'].apply(lambda x: 1 if x != 4 else 0)
    df_iadl.loc[df_iadl.question == '<center>Shopping</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Shopping</center>', 'score'].apply(lambda x: 1 if x == 1 else 0)
    df_iadl.loc[df_iadl.question == '<center>Foodpreparation</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Foodpreparation</center>', 'score'].apply(lambda x: 1 if x == 1 else 0)
    df_iadl.loc[df_iadl.question == '<center>Housekeeping</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Housekeeping</center>', 'score'].apply(lambda x: 1 if x != 5 else 0)
    df_iadl.loc[df_iadl.question == '<center>Laundry</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Laundry</center>', 'score'].apply(lambda x: 1 if x != 3 else 0)
    df_iadl.loc[df_iadl.question == '<center>Modeoftransportation</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Modeoftransportation</center>', 'score'].apply(lambda x: 1 if x <= 3 else 0)
    df_iadl.loc[df_iadl.question == '<center>Responsibilityforownmedication</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Responsibilityforownmedication</center>', 'score'].apply(lambda x: 1 if x == 1 else 0)
    df_iadl.loc[df_iadl.question == '<center>Abilitytohandlefinances</center>', 'score'] = df_iadl.loc[df_iadl.question == '<center>Abilitytohandlefinances</center>', 'score'].apply(lambda x: 1 if x != 3 else 0)

    scores = [None] * len(df_iadl.user_id.unique())

    for count,id in enumerate(df_iadl.user_id.unique()):
        
        
        df_raw_temp = df_iadl.copy().loc[df_iadl.user_id == id, :]
        df_raw_temp.reset_index(drop=True, inplace=True) 
        scores[count] = sum(df_raw_temp.score)
        
    df_iadl_summary = pd.DataFrame({"user_id":df_iadl.user_id.unique(), "IADL":scores})
    
        
    df_iadl_summary.to_csv(f'../{output_clean_folder}{folder_structure[0]}/{questionnaire_name}{clean_file_extension}{data_format}', index=False)
    
    return df_iadl_summary
    
    

def demographics_preproc(root_path, merged_data_folder, questionnaire_name, folder_structure, data_format, clean_file_extension, output_clean_folder):
    
    os.chdir(root_path + merged_data_folder)
    try:
        df_dem_summary = pd.read_csv(f'.{folder_structure[0]}/{questionnaire_name}{data_format}', low_memory=False)
        
        df_dem = pd.read_csv(f'.{folder_structure[1]}/{questionnaire_name}_raw{data_format}', low_memory=False)
        
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
    english = df_dem.groupby(['question']).get_group('<center>HowwouldyourateyourproficiencyinEnglish?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    depression = df_dem.groupby(['question']).get_group('<center>Areyoucurrentlytakinganymedicationfordepression?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    anxiety = df_dem.groupby(['question']).get_group('<center>Areyoucurrentlytakinganymedicationforanxiety?</center>').loc[:,['response','user_id']].reset_index(drop=True)
    
    # Clean each variable
    
    age.response = pd.to_numeric(age.response)
    age.loc[age.response < 18,'response'] = np.nan
    
    gender.response = gender.response.replace(['62','41'],np.nan)
    gender.replace(['Male','Female'],[0,1], inplace=True)

    education.response = education.response.replace('1',np.nan)
    education.response = education.response.replace('Secondary/HighSchoolDiploma','Secondary/HighSchool-A-levels')
    education.response = education.response.replace('Primary/ElementarySchool','SecondarySchool-GCSE')
    education.replace(['SecondarySchool-GCSE','Secondary/HighSchool-A-levels','ProfessionalDegree','Bachelor\'sDegree','Master\'sDegree','PhD'],[0,1,1,2,3,3], inplace=True)
                
    device = pd.DataFrame()        
    device['response'] = df_dem_summary.loc[:,['os']]
    device['user_id'] = df_dem_summary.loc[:,['user_id']]
    device.response.replace(['iOS','Mac OS X','Windows','Android'],[0,1,2,0], inplace=True)

    
    english.response.replace({'3': 1, '4': 1, 'No': np.nan, '2':1,'1':0}, inplace=True)
    depression.response.replace({'No':0, 'Yes':1, 'SKIPPED':0}, inplace=True)   
    anxiety.response.replace({'No':0, 'Yes':1, 'SKIPPED':0}, inplace=True)
    
    age.drop_duplicates(subset="user_id",keep='last',inplace=True)
    gender.drop_duplicates(subset="user_id", keep='last',inplace=True)
    education.drop_duplicates(subset="user_id", keep='last',inplace=True)
    device.drop_duplicates(subset="user_id", keep='last',inplace=True)
    english.drop_duplicates(subset="user_id", keep='last',inplace=True)
    depression.drop_duplicates(subset="user_id", keep='last',inplace=True)
    anxiety.drop_duplicates(subset="user_id", keep='last',inplace=True)
    
    age.rename(columns={'response':'age'}, inplace=True)
    gender.rename(columns={'response':'gender'}, inplace=True)
    education.rename(columns={'response':'education'}, inplace=True)
    device.rename(columns={'response':'device'}, inplace=True)
    english.rename(columns={'response':'english'}, inplace=True)
    depression.rename(columns={'response':'depression'}, inplace=True)
    anxiety.rename(columns={'response':'anxiety'}, inplace=True)
    
    age.dropna(inplace=True)
    gender.dropna(inplace=True)
    education.dropna(inplace=True)
    device.dropna(inplace=True)
    english.dropna(inplace=True)
    depression.dropna(inplace=True)
    anxiety.dropna(inplace=True)
              
    # Merge and format
    
    pat_demographics = age.merge(gender,on='user_id').merge(education,on='user_id').merge(device,on='user_id').merge(english,on='user_id').merge(depression,on='user_id').merge(anxiety,on='user_id')
    pat_demographics.education = pat_demographics.education.astype(int)
    pat_demographics['age2'] = pat_demographics.age**2
    pat_demographics['dyslexia'] = 0
    
    # Update missing or inconsistent data
    
    pat_demographics.index = pat_demographics.user_id
    pat_demographics.loc['00009-session1-versionA',:] = pat_demographics.loc['ic3study00009-session2-versionA',:].values.tolist()
    pat_demographics.loc['ic3study00019-session1-versionA',:] = pat_demographics.loc['ic3study00019-session2-versionB',:].values.tolist()
    pat_demographics.loc['ic3study00035-session2-versionB',:] = pat_demographics.loc['ic3study00035-session1-versionA',:].values.tolist()
    pat_demographics.loc['ic3study00041-session2-versionB',:] = pat_demographics.loc['ic3study00041-session1-versionA',:].values.tolist()
    pat_demographics.loc['ic3study00041-session2-versionB','device'] = 1 
    pat_demographics.loc['ic3study00050-session1-versionA',:] = pat_demographics.loc['ic3study00050-session2-versionB',:].values.tolist()
    pat_demographics.loc['ic3study00050-session1-versionA','device'] = 1
    pat_demographics.loc['ic3study00051-session1-versionA',:] = pat_demographics.loc['00051-session2-versionB',:].values.tolist()
    pat_demographics.loc['ic3study00051-session1-versionA','device'] = 1
    pat_demographics.loc['ic3study00095-session1-versionA',:] = pat_demographics.loc['ic3study00095-session2-versionB',:].values.tolist()
    pat_demographics.drop(columns='user_id', inplace=True)
    pat_demographics.reset_index(drop=False,inplace=True)
    
    pat_demographics.device = pat_demographics.device.astype(float).astype(int)
    pat_demographics.education = pat_demographics.education.astype(float).astype(int)
    
    one_hot_encoded_data = pd.get_dummies(pat_demographics, columns = ['device', 'education'])
    one_hot_encoded_data.rename(columns={'education_1':'education_Alevels', 'education_2':'education_bachelors','education_3':'education_postBachelors'}, inplace=True)
    one_hot_encoded_data.rename(columns={'device_1':'device_tablet', 'device_0':'device_phone'}, inplace=True)
    one_hot_encoded_data.rename(columns={'english':'english_secondLanguage'}, inplace=True)
    one_hot_encoded_data.replace({True:1, False:0}, inplace=True)
    one_hot_encoded_data.loc[:,'gender':'education_postBachelors'] = one_hot_encoded_data.loc[:,'gender':'education_postBachelors'] -0.5
    one_hot_encoded_data.drop(columns=['device_2','education_0'], inplace=True)
    
    # Save 
    
    one_hot_encoded_data.to_csv(f'../{output_clean_folder}{folder_structure[0]}/{questionnaire_name}{clean_file_extension}{data_format}', index=False)
    
    return one_hot_encoded_data

  
def remove_general_outliers(root_path, merged_data_folder, task_name, data_format, folder_structure):
    
    os.chdir(root_path + merged_data_folder)
    
    try:
        df = pd.read_csv(f'.{folder_structure[0]}/{task_name}{data_format}', low_memory=False)

        df_raw = pd.read_csv(f'.{folder_structure[1]}/{task_name}_raw{data_format}', low_memory=False)
    except:
        print(f'Error in loading {task_name}. File might not exist.')
        return None,None

    df.drop_duplicates(subset=['user_id'],keep="last", inplace=True)
    df.drop(columns=['Unnamed: 0','type','sequenceObj', 'dynamicDifficulty'], inplace=True)
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


    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw

def pear_cancellation_preproc(df,df_raw):
    
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
        if df_raw_temp.empty:
            continue
        ids[count] = id        
        
        errors[count] = (sum(df_raw_temp.correct == False))
        meanRTs[count] = np.nanmean(df_raw_temp.RT)
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan  
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct).any() else np.nan
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
    
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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw  


def srt_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<180,'RT'] = np.nan
    df_raw.loc[df_raw.RT>1000,'RT'] = np.nan


    meanRTs =[None] * len(df.user_id)
    medianRTs =[None] * len(df.user_id)
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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def motor_control_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>10000,'RT'] = np.nan

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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

def taskrecall_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
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
    
    df.loc[df.SummaryScore == 9, "SummaryScore"] = 8
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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def gesture_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<1500,'RT'] = np.nan
    df_raw.loc[df_raw.RT>25000,'RT'] = np.nan

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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def rule_learning_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan

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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    
    return df,df_raw  


  
def oddoneout_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  

    
def comprehension_preproc(df,df_raw):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
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
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
        
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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def trailmaking_preproc(df,df2,df3,df_raw,df_raw2,df_raw3, task_name):
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan
    
    df_raw2.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw2.loc[df_raw.RT>15000,'RT'] = np.nan

    df_raw3.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw3.loc[df_raw.RT>15000,'RT'] = np.nan


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
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
    
        medianRTs[count] = np.nanmedian(df_raw2_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
        
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
        meanErrorRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        meanCorrRTs[count] = np.nanmean(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
        
        medianRTs[count] = np.nanmedian(df_raw3_temp.RT)
        medianErrorRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == False].RT) if (~df_raw_temp.correct.dropna()).any() else np.nan
        medianCorrRTs[count] = np.nanmedian(df_raw_temp[df_raw_temp.correct == True].RT) if (df_raw_temp.correct.dropna()).any() else np.nan
        
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

    df3.reset_index(drop=True,inplace=True)
    df_raw3 = df_raw3[df_raw3.user_id.isin(df3.user_id)]
    df_raw3.reset_index(drop=True,inplace=True)

    df = df.loc[:, ['user_id', 'totalCorrect', 'totalIncorrect','meanCorrRT','medianCorrRT']]
    df2 = df2.loc[:, ['user_id', 'totalCorrect', 'totalIncorrect','meanCorrRT','medianCorrRT']]
    df3.rename(columns={"totalCorrect": "totalCorrect_level3", "totalIncorrect": "totalIncorrect_level3",'meanCorrRT':'meanCorrRT_level3','medianCorrRT':'medianCorrRT_level3'}, inplace=True)
    df.rename(columns={"totalCorrect": "totalCorrect_level1", "totalIncorrect": "totalIncorrect_level1",'meanCorrRT':'meanCorrRT_level1','medianCorrRT':'medianCorrRT_level1'}, inplace=True)
    df2.rename(columns={"totalCorrect": "totalCorrect_level2", "totalIncorrect": "totalIncorrect_level2",'meanCorrRT':'meanCorrRT_level2','medianCorrRT':'medianCorrRT_level2'}, inplace=True)
    df_temp = df.merge(df2, on='user_id', how='outer').merge(df3, on='user_id', how='outer')
    
    df_temp['switchCostAccuracy'] = df_temp.totalCorrect_level3 - df_temp.totalCorrect_level2 - df_temp.totalCorrect_level1
    df_temp['switchCostErrors'] = df_temp.totalIncorrect_level3 - (df_temp.totalIncorrect_level2 + df_temp.totalIncorrect_level1)/2 
    df_temp['switchCostErrors'].fillna(6,inplace=True)
    
    df_temp['switchCostMedianCorrRT'] = df_temp.medianCorrRT_level3 - (df_temp.medianCorrRT_level2 + df_temp.medianCorrRT_level1)/2
    df_temp['switchCostMeanCorrRT'] = df_temp.meanCorrRT_level3 - (df_temp.meanCorrRT_level2 + df_temp.meanCorrRT_level1)/2
    df_temp.loc[df_temp['switchCostErrors'] <0, 'switchCostErrors'] = 0
    df_temp.loc[df_temp['switchCostAccuracy'] <0, 'switchCostAccuracy'] = 0
    df_temp.loc[df_temp['switchCostMedianCorrRT'] <0, 'switchCostMedianCorrRT'] = 0
    df_temp.loc[df_temp['switchCostMeanCorrRT'] <0, 'switchCostMeanCorrRT'] = 0
    
    df_temp["SummaryScore"] = df_temp.switchCostErrors
    df_temp["trailAll"] = df_temp.switchCostErrors

    df_temp_raw = pd.concat([df_raw,df_raw2,df_raw3],ignore_index=True,axis=0)   
    df_temp_raw.reset_index(drop=True,inplace=True)
    
    df_temp.taskID = task_name
    df_temp.reset_index(drop=True,inplace=True)
    
    return df_temp,df_temp_raw  