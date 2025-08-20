"""
Last updated on 20th of March 2025
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

def main_preprocessing(
    root_path,
    list_of_tasks,
    list_of_questionnaires,
    list_of_speech,
    remote_data_folders='/data_ic3online_cognition',
    supervised_data_folders=['/data_healthy_v1', '/data_healthy_v2'],
    folder_structure=['/summary_data', '/trial_data', '/speech'],
    output_clean_folder='/data_healthy_cleaned',
    merged_data_folder='/data_healthy_combined',
    clean_file_extension='_cleaned',
    data_format='.csv',
    append_data = False
):
    
    """
    Main wrapper function that cleans normative data, creates inclusion criteria,
    merges data across sites, and preprocesses tasks and questionnaires.

    Parameters
    ----------
    root_path : str
        The root directory for processing.
    list_of_tasks : list of str
        List of task names to process.
    list_of_questionnaires : list of str
        List of questionnaire names to process.
    list_of_speech : list of str
        List of speech file identifiers.
    remote_data_folders : str, optional
        Remote data folder path (default: '/data_ic3online_cognition').
    supervised_data_folders : list of str, optional
        List of supervised data folder paths (default: ['/data_healthy_v1', '/data_healthy_v2']).
    folder_structure : list of str, optional
        Folder structure used (default: ['/summary_data', '/trial_data', '/speech']).
    output_clean_folder : str, optional
        Output folder for cleaned data (default: '/data_healthy_cleaned').
    merged_data_folder : str, optional
        Folder for merged data (default: '/data_healthy_combined').
    clean_file_extension : str, optional
        Suffix to append to cleaned files (default: '_cleaned').
    data_format : str, optional
        Data file format (default: '.csv').
    append_data : bool, optional
        Append data to existing file or overwrite existing excel file (default is False).

    """
    
    print('Starting preprocessing...')

    # Build an absolute path for root_path
    root_path = os.path.abspath(root_path)
    
    # Ensure merged data folder exists (strip leading '/' for relative joining)
    merged_data_dir = os.path.join(root_path, merged_data_folder.lstrip('/'))
    ensure_directory(merged_data_dir)

    # ----- Cleaning normative data and compiling inclusion criteria -----
    
    print('Cleaning normative data. Creating inclusion criteria...', end="", flush=True)

    inclusion_criteria = create_inclusion_criteria(root_path, remote_data_folders, supervised_data_folders, folder_structure)
    # Save inclusion criteria
    #inclusion_criteria_path = os.path.join(merged_data_dir, 'inclusion_criteria.csv')
    #inclusion_criteria.to_csv(inclusion_criteria_path, index=False)
    print('Done')

    # ----- Merging data across sites -----
    
    print('Merging data across sites...', end="", flush=True)
    if (supervised_data_folders is None):
        merged_data_folder = remote_data_folders
    else:
        list_of_tasks = merge_control_data_across_sites(
            root_path, folder_structure, supervised_data_folders, remote_data_folders,
            list_of_tasks, list_of_questionnaires, list_of_speech, data_format, merged_data_folder
        )
    print('Done')
    
    # ----- Pre-processing task data -----
    
    if list_of_tasks:
        
        for task_name in list_of_tasks:
            
            # Process a single task by first removing general outliers and then applying a task-specific preprocessing
            print(f'Pre-processing {task_name}...', end="", flush=True)
            df, df_raw = remove_general_outliers(root_path, merged_data_folder, task_name, inclusion_criteria, data_format, folder_structure)
            
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
            
            output_preprocessed_data(df,df_raw, root_path, output_clean_folder, folder_structure,  clean_file_extension, data_format, append_data)
            print('Done')
    else:
        print('No tasks were provided.') 
        
    # ----- Cleaning questionnaire data -----    
        
    if list_of_questionnaires:
        for questionnaire_name in list_of_questionnaires:
            process_questionnaire(questionnaire_name, root_path, merged_data_folder, inclusion_criteria,
                                  folder_structure, data_format, clean_file_extension, output_clean_folder, list_of_tasks,append_data)
    else:
        print('No questionnaires were provided.')
                   
    print('Preprocessing complete.')      


def ensure_directory(path):
    """Checks if directory exists, creates it if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def create_inclusion_criteria(root_path, remote_data_folders, supervised_data_folders, folder_structure):
    """
    Create inclusion criteria by performing outlier detection on remote and supervised data.

    This function performs outlier detection on data from both remote and supervised settings using
    designated outlier detection functions. It then concatenates the results into a single pandas
    DataFrame which represents the inclusion criteria.

    Parameters
    ----------
    root_path : str
        The base directory path where the data folders are located.
    remote_data_folders : list of str
        A list of folder names that contain remote data.
    supervised_data_folders : list of str
        A list of folder names that contain supervised data.
    folder_structure : dict
        A dictionary representing the structure of folders used to organize the data. The expected
        format and keys should correspond with what the outlier detection functions require.

    Returns
    -------
    inclusion_criteria : pandas.DataFrame
        A DataFrame containing the inclusion criteria generated by combining the outlier detection
        results from remote and supervised data.

    See Also
    --------
    general_outlier_detection_remoteSetting : Function that detects outliers in remote data.
    general_outlier_detection_supervisedSetting : Function that detects outliers in supervised data.
    """
    
    if remote_data_folders is not None:
        ids_remote = general_outlier_detection_remoteSetting(
            root_path, remote_data_folders, folder_structure
        )
        inclusion_criteria = ids_remote
    if supervised_data_folders is not None:
        ids_supervised = general_outlier_detection_supervisedSetting(
            root_path, supervised_data_folders, folder_structure
        )
        inclusion_criteria = ids_supervised
        
    if (remote_data_folders is not None) & (supervised_data_folders is not None):
        inclusion_criteria = pd.concat([ids_remote, ids_supervised], ignore_index=True)
    inclusion_criteria.reset_index(drop=True, inplace=True)

    return inclusion_criteria


def process_questionnaire(questionnaire_name, root_path, merged_data_folder, inclusion_criteria, folder_structure, data_format, clean_file_extension, output_clean_folder, list_of_tasks, append_data):
    """
    Process a questionnaire using a match-case structure.

    This function processes a questionnaire by matching the given questionnaire name to a specific 
    preprocessing routine. Currently, it handles the case for 'q_IC3_demographics' where it calls 
    functions to preprocess demographics data and subsequently combines demographics data with 
    cognition data. If the questionnaire name does not match any recognized case, it outputs a 
    notification indicating that no specific preprocessing function is available.

    Parameters
    ----------
    questionnaire_name : str
        The name of the questionnaire to be processed.
    root_path : str
        The base directory path where the data folders are located.
    merged_data_folder : str
        The folder name that contains the merged data files.
    inclusion_criteria : pandas.DataFrame
        A DataFrame containing the inclusion criteria used for filtering or processing data.
    folder_structure : dict
        A dictionary defining the organization of data folders.
    data_format : str
        The format of the data files (e.g., 'csv', 'xlsx').
    clean_file_extension : str
        The file extension used for the cleaned data files.
    output_clean_folder : str
        The directory where the processed (cleaned) data files will be saved.
    list_of_tasks : list
        A list of tasks or steps to be executed during the processing pipeline.

    Returns
    -------
    None

    See Also
    --------
    demographics_preproc : Function to preprocess demographics data.
    combine_demographics_and_cognition : Function to combine demographics data with cognition data.
    """
    
    print(f'Pre-processing {questionnaire_name}...', end="", flush=True)
    match questionnaire_name:
        case 'q_IC3_demographics':
            df_demographics = demographics_preproc(
                root_path, merged_data_folder, output_clean_folder, questionnaire_name, inclusion_criteria, folder_structure, data_format, clean_file_extension, append_data
            )
            combine_demographics_and_cognition(
                root_path, output_clean_folder, folder_structure, list_of_tasks, df_demographics, clean_file_extension, data_format, append_data
            )
            print('Done')
        case _:
            print(f'\nQuestionnaire {questionnaire_name} does not have a specific preprocessing function.')


def general_outlier_detection_remoteSetting(root_path, remote_data_folders, folder_structure, 
                                              screening_list=None):
    """
    Detect outliers in remote setting data by loading multiple screening questionnaires,
    cleaning and merging them, and then applying exclusion criteria.

    Parameters
    ----------
    root_path : str
        The root directory.
    remote_data_folder : str
        Remote data folder name (e.g. '/data_ic3online_cognition').
    folder_structure : list of str
        List containing folder names; the first element is used to locate summary data.
    screening_list : list of str, optional
        List of questionnaire filenames. Defaults to:
        ['q_IC3_demographicsHealthy_questionnaire.csv',
         'q_IC3_metacog_questionnaire.csv',
         'IC3_Orientation.csv',
         'IC3_PearCancellation.csv'].

    Returns
    -------
    pandas.Series
        A Series of user IDs that pass all exclusion criteria.
    """
    
    if screening_list is None:
        screening_list = [
            'q_IC3_demographicsHealthy_questionnaire.csv',
            'q_IC3_metacog_questionnaire.csv',
            'IC3_Orientation.csv',
            'IC3_PearCancellation.csv'
        ]
        
    # Build the base path for the remote data
    base_path = os.path.join(root_path, remote_data_folders.lstrip('/'), folder_structure[0].lstrip('/'))
    
    # Load screening questionnaires from file paths
    df_dem    = pd.read_csv(os.path.join(base_path, screening_list[0]), low_memory=False)
    df_cheat  = pd.read_csv(os.path.join(base_path, screening_list[1]), low_memory=False)
    df_orient = pd.read_csv(os.path.join(base_path, screening_list[2]), low_memory=False)
    df_pear   = pd.read_csv(os.path.join(base_path, screening_list[3]), low_memory=False)
    
    # Define a helper to remove duplicate and missing user IDs
    def clean_df(df):
        return df.drop_duplicates(subset=['user_id'], keep="first").dropna(subset=['user_id']).reset_index(drop=True)
    
    df_dem    = clean_df(df_dem)
    df_cheat  = clean_df(df_cheat)
    df_orient = clean_df(df_orient)
    df_pear   = clean_df(df_pear)
        
    # Keep only users present in both demographics and Pear Cancellation data
    common_ids = set(df_dem['user_id']) ^ set(df_pear['user_id'])

    for df in (df_dem, df_orient, df_pear, df_cheat):
        df.drop(df[df['user_id'].isin(common_ids)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    # Clean the questionnaire data
    df_dem['Q30_R'] = df_dem['Q30_R'].replace("No", "SKIPPED").replace("SKIPPED", 999999).astype(float)
    for col in ["Q2_S", "Q3_S"]:
        df_cheat[col].replace([0, 1], np.nan, inplace=True)
    df_cheat.dropna(subset=["Q2_S", "Q3_S"], inplace=True)
    df_cheat.reset_index(drop=True, inplace=True)
        
    # Remove users who fail screening tests based on SummaryScore criteria
    # Vectorized approach: identify user_ids that meet both failure conditions.
    fail_orient = set(df_orient.loc[df_orient['SummaryScore'] < 3, 'user_id'])
    fail_pear   = set(df_pear.loc[df_pear['SummaryScore'] <= 0.80, 'user_id'])
    failed_ids  = fail_orient & fail_pear    
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(failed_ids)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f'We removed {len(failed_ids)} people who failed both Orientation and Pear Cancellation, from all tasks.')
    
    # Remove users not neurologically healthy (using demographics responses)
    unhealthy_mask = ((df_dem['Q12_R'] != "SKIPPED") | (df_dem['Q14_R'] != "SKIPPED") |
                      (df_dem['Q30_R'] <= 60) | (df_dem['Q1_R'] < 40))
    unhealthy_mask = df_dem.loc[unhealthy_mask,'user_id']
    neuro_removed = ((df_dem['Q12_R'] != "SKIPPED") | (df_dem['Q14_R'] != "SKIPPED")).sum()
    dementia_removed = (df_dem['Q30_R'] <= 60).sum()
    age_removed = (df_dem['Q1_R'] < 40).sum()
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(unhealthy_mask)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f'We removed {neuro_removed} who indicated neurological disorder, '
          f'{dementia_removed} with a history of dementia and {age_removed} who are younger than 40.')
    
    # Remove users who self-reported lack of engagement ("cheating")
    cheating_mask = df_pear['user_id'].isin(df_cheat['user_id'])
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(df_cheat['user_id'])].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    print(f'We removed {cheating_mask.sum()} people who cheated.')
    
    # Return the final cleaned user IDs from df_pear as a Series
    cleaned_ids_remote = df_pear['user_id']
    
    return cleaned_ids_remote


def general_outlier_detection_supervisedSetting(root_path, supervised_data_folders, folder_structure, 
                                                  screening_list=None):
    """
    Detect outliers in supervised setting data by loading two timepoints of screening questionnaires,
    cleaning and merging them, and then applying exclusion criteria.

    Parameters
    ----------
    root_path : str
        The root directory.
    supervised_data_folders : list of str
        List containing folder names for the two timepoints (e.g. [folder_v1, folder_v2]).
    folder_structure : list of str
        List containing folder names; the first element is used to locate summary data.
    screening_list : list of str, optional
        List of questionnaire filenames. Defaults to:
        ['q_IC3_demographics_questionnaire.csv', 'IC3_Orientation.csv', 'IC3_PearCancellation.csv'].

    Returns
    -------
    pandas.Series
        A Series of user IDs that pass all exclusion criteria.
    """
    if screening_list is None:
        screening_list = [
            'q_IC3_demographics_questionnaire.csv',
            'IC3_Orientation.csv',
            'IC3_PearCancellation.csv'
        ]
    
    def load_timepoint_data(data_folder):
        """Helper function to load data for a given timepoint."""
        base_path = os.path.join(root_path, data_folder.lstrip('/'), folder_structure[0].lstrip('/'))
        df_dem    = pd.read_csv(os.path.join(base_path, screening_list[0]), low_memory=False)
        df_orient = pd.read_csv(os.path.join(base_path, screening_list[1]), low_memory=False)
        df_pear   = pd.read_csv(os.path.join(base_path, screening_list[2]), low_memory=False)
        return df_dem, df_orient, df_pear
    
    # Load data for both timepoints
    df_dem_tp1, df_orient_tp1, df_pear_tp1 = load_timepoint_data(supervised_data_folders[0])
    df_dem_tp2, df_orient_tp2, df_pear_tp2 = load_timepoint_data(supervised_data_folders[1])
    
    # Concatenate the two timepoints
    df_dem    = pd.concat([df_dem_tp1, df_dem_tp2], ignore_index=True)
    df_orient = pd.concat([df_orient_tp1, df_orient_tp2], ignore_index=True)
    df_pear   = pd.concat([df_pear_tp1, df_pear_tp2], ignore_index=True)
    
    def clean_df(df):
        """Remove duplicates and missing user IDs, then reset index."""
        return df.drop_duplicates(subset=['user_id'], keep='first') \
                 .dropna(subset=['user_id']) \
                 .reset_index(drop=True)
    
    # Clean dataframes
    df_dem    = clean_df(df_dem)
    df_orient = clean_df(df_orient)
    df_pear   = clean_df(df_pear)
    
    # Retain only common user IDs across all dataframes
    common_ids = set(df_dem['user_id']) ^ set(df_pear['user_id'])
    
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(common_ids)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Remove users who fail screening tests based on SummaryScore criteria
    # Vectorized approach: identify user_ids that meet both failure conditions.
    fail_orient = set(df_orient.loc[df_orient['SummaryScore'] < 3, 'user_id'])
    fail_pear   = set(df_pear.loc[df_pear['SummaryScore'] <= 0.80, 'user_id'])
    failed_ids  = fail_orient & fail_pear    
    
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(failed_ids)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    print(f'We removed {len(failed_ids)} people who failed both Orientation and Pear Cancellation, from all tasks.')

    # Remove users not neurologically healthy (using demographics responses)
    unhealthy_mask = ((df_dem['Q12_R'] != "SKIPPED") | (df_dem['Q14_R'] != "SKIPPED") |
                    (df_dem['Q1_R'] < 40))
    unhealthy_mask = df_dem.loc[unhealthy_mask,'user_id']
    neuro_removed = ((df_dem['Q12_R'] != "SKIPPED") | (df_dem['Q14_R'] != "SKIPPED")).sum()
    age_removed = (df_dem['Q1_R'] < 40).sum()
    
    for df in (df_dem, df_orient, df_pear):
        df.drop(df[df['user_id'].isin(unhealthy_mask)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f'We removed {neuro_removed} who indicated neurological disorder, '
          f'{age_removed} who are younger than 40.')
    
    # Return the final cleaned user IDs from df_pear as a Series
    cleaned_ids_supervised = df_pear['user_id']
    
    return cleaned_ids_supervised


def merge_control_data_across_sites(root_path, folder_structure, supervised_data_folders,
                                    remote_data_folders, list_of_tasks, list_of_questionnaires,
                                    list_of_speech, data_format, merged_data_folder):
    """
    Merge control data across sites from clinical tests, speech, and questionnaires.

    This function creates a consolidated dataset by merging control data gathered from 
    multiple sources (clinical tests, speech, and questionnaires) across various sites. 
    It builds the output base directory under the specified merged data folder, ensures that 
    the required directory structure exists, and then sequentially calls dedicated functions 
    to merge clinical tests, speech data, and questionnaire data.

    Parameters
    ----------
    root_path : str
        The base directory path where the data is located.
    folder_structure : iterable of str
        An iterable (e.g., list or keys from a dict) containing folder names that define the 
        required directory structure.
    supervised_data_folders : list of str
        A list of folder names containing supervised data.
    remote_data_folders : list of str
        A list of folder names containing remote data.
    list_of_tasks : list of str
        A list of tasks or operations to be performed during the merging process.
    list_of_questionnaires : list of str
        A list of questionnaire identifiers whose data is to be merged.
    list_of_speech : list of str
        A list of identifiers or folder names for speech data to be merged.
    data_format : str
        The file format of the data files (e.g., 'csv', 'xlsx').
    merged_data_folder : str
        The name of the folder where the merged data will be stored. This will be appended 
        to the root path to create the output base directory.

    Returns
    -------
    None

    See Also
    --------
    ensure_directory : Function to create a directory if it does not exist.
    merge_clinical_tests : Function that merges clinical test data.
    merge_speech_data : Function that merges speech data.
    merge_questionnaire_data : Function that merges questionnaire data.
    """
    
    # Build output base folder and ensure directory structure exists
    output_base = os.path.join(root_path, merged_data_folder.strip("/"))
    ensure_directory(output_base)
    
    for folder in folder_structure:
        ensure_directory(os.path.join(output_base, folder.strip("/")))
        
    # Merge clinical test data
    merge_clinical_tests(root_path, folder_structure, supervised_data_folders,
                                         remote_data_folders, list_of_tasks, data_format, output_base)
    
    # Merge speech data
    merge_speech_data(root_path, folder_structure, supervised_data_folders, list_of_speech,  data_format, output_base)
    
    # Merge questionnaire data
    merge_questionnaire_data(root_path, folder_structure, supervised_data_folders, remote_data_folders,
                             list_of_questionnaires,  data_format, output_base)
    
    return list_of_tasks


def load_task_data(root, data_folder, subfolder, task, data_format, raw=False, version_suffix=''):
    """
    Load task data from a CSV file.

    Constructs the file path based on the provided directory components and loads the corresponding
    CSV file as a pandas DataFrame. The function supports an optional flag to load raw data and an
    optional version suffix to accommodate special naming conventions.

    Parameters
    ----------
    root : str
        The root directory where the data is stored.
    data_folder : str
        The name of the data folder (e.g., 'supervised' or 'remote').
    subfolder : str
        The subfolder name within the data folder (e.g., 'summary' or 'raw').
    task : str
        The task name used as part of the file path.
    data_format : str
        The file extension, such as '.csv'.
    raw : bool, optional
        If True, appends '_raw' to the filename to load the raw file. Default is False.
    version_suffix : str, optional
        An optional suffix to append to the filename for special cases (e.g., '_v2' for versioning).
        Default is an empty string.

    Returns
    -------
    pandas.DataFrame
        The loaded data as a pandas DataFrame.
    """
    base = os.path.join(root, data_folder.strip("/"), subfolder.strip("/"), task)
    suffix = "_raw" if raw else ""
    filename = f"{base}{version_suffix}{suffix}{data_format}"

    return pd.read_csv(filename, low_memory=False)

def merge_and_save(df_list, output_file):
    """
    Concatenate a list of DataFrames and save the merged DataFrame to a CSV file.

    Merges multiple pandas DataFrames into a single DataFrame, prints the total number of rows
    in the merged DataFrame, and saves the resulting DataFrame to the specified output file.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        A list of DataFrames to be concatenated.
    output_file : str
        The full file path (including the filename) where the merged DataFrame will be saved.

    """
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)

def merge_clinical_tests(root, folder_structure, supervised_data_folders, remote_data_folder, tasks, data_format, output_base):
    
    """
    Merge clinical test data across sources.

    This function merges clinical test data from multiple sources, including two supervised sources
    and one remote source. It handles special cases such as the 'IC3_i4i_IDED' task and temporarily 
    adds additional tasks for 'IC3_NVtrailMaking'. For each task, it loads both processed and raw 
    data from the respective sources, merges them, and saves the resulting files in the designated 
    output directories.

    Parameters
    ----------
    root : str
        The root directory where the data files are located.
    folder_structure : list or tuple of str
        A two-element sequence containing the names of the subfolders for processed and raw data 
        respectively (e.g., ['processed_folder', 'raw_folder']).
    supervised_data_folders : list of str
        A list containing the names of the supervised data source folders. The first element corresponds 
        to the first supervised source and the second to the second supervised source.
    remote_data_folder : str
        The folder name for the remote data source.
    tasks : list of str
        A list of task names for which the data should be merged.
    data_format : str
        The file extension or data format (e.g., '.csv').
    output_base : str
        The base output directory where the merged files will be saved. The output paths are constructed 
        using the provided folder_structure.

    See Also
    --------
    load_task_data : Function to load task data from a CSV file.
    merge_and_save : Function to merge a list of DataFrames and save them to a CSV file.
    """
    
    # Use a copy of the tasks list to add temporary tasks
    tasks_to_merge = tasks.copy()
    if 'IC3_NVtrailMaking' in tasks_to_merge:
        tasks_to_merge += ['IC3_NVtrailMaking2', 'IC3_NVtrailMaking3']

    for task in tasks_to_merge:
        # Supervised source 1
        df_v1 = load_task_data(root, supervised_data_folders[0], folder_structure[0], task, data_format, raw=False)
        df_v1_raw = load_task_data(root, supervised_data_folders[0], folder_structure[1], task, data_format, raw=True)
        
        # Supervised source 2 (with special case for IDED)
        if task == "IC3_i4i_IDED":
            df_v2 = load_task_data(root, supervised_data_folders[1], folder_structure[0], task, data_format, raw=False, version_suffix='2')
            df_v2_raw = load_task_data(root, supervised_data_folders[1], folder_structure[1], task, data_format, raw=True, version_suffix='2')
        else:
            df_v2 = load_task_data(root, supervised_data_folders[1], folder_structure[0], task, data_format, raw=False)
            df_v2_raw = load_task_data(root, supervised_data_folders[1], folder_structure[1], task, data_format, raw=True)
        
        summary_out = os.path.join(output_base, folder_structure[0].strip("/"), f"{task}{data_format}")
        raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
        if remote_data_folder is not None:
            # Remote data source
            df_cog = load_task_data(root, remote_data_folder, folder_structure[0], task, data_format, raw=False)
            df_cog_raw = load_task_data(root, remote_data_folder, folder_structure[1], task, data_format, raw=True)
            # Merge all three sources
            merge_and_save([df_v1, df_v2, df_cog], summary_out)
            merge_and_save([df_v1_raw, df_v2_raw, df_cog_raw],raw_out)
            
        else:
            # Merge the two supervised sources
            merge_and_save([df_v1, df_v2], summary_out)
            merge_and_save([df_v1_raw, df_v2_raw],raw_out)
            

        # Optionally log progress: print(f'Merged clinical test data for {task}')

def merge_speech_data(root, folder_structure, supervised_data_folders, list_of_speech,  data_format, output_base):
    """
    Merge speech data from two supervised sources.

    This function merges speech data collected from two supervised data sources. For each speech 
    task, it loads the raw data from both sources, concatenates the results, and saves the merged 
    data into the designated output directory.

    Parameters
    ----------
    root : str
        The root directory where the data is stored.
    folder_structure : list or tuple of str
        A two-element sequence containing the names of the subfolders for processed and raw data.
        In this function, the raw data folder is used.
    supervised_data_folders : list of str
        A list containing the names of the supervised data folders. There should be at least two 
        elements corresponding to the two sources.
    list_of_speech : list of str
        A list of speech task names to be merged.
    data_format : str
        The file extension or data format (e.g., '.csv').
    output_base : str
        The base directory where the merged speech data will be saved.

    See Also
    --------
    load_task_data : Function to load task data from a CSV file.
    merge_and_save : Function to merge a list of DataFrames and save them to a CSV file.
    """
    
    for task in list_of_speech:
        df_v1_raw = load_task_data(root, supervised_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True)
        df_v2_raw = load_task_data(root, supervised_data_folders[1], folder_structure[1].strip("/"), task, data_format, raw=True)
        
        raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
        merge_and_save([df_v1_raw, df_v2_raw], raw_out)
        
        # Optionally log progress: print(f'Merged speech data for {task}')

def merge_questionnaire_data(root, folder_structure, supervised_data_folders, remote_data_folder, list_of_questionnaires,  data_format, output_base):
    """
    Merge questionnaire data from multiple sources.

    This function merges questionnaire data from both supervised and remote sources. It handles 
    a special case for demographics data by using distinct version suffixes when loading files, and 
    processes other questionnaires using a standard naming convention. For each questionnaire, both 
    processed and raw data are merged and saved to the designated output directories.

    Parameters
    ----------
    root : str
        The root directory where the data files are located.
    folder_structure : list or tuple of str
        A two-element sequence containing the names of the subfolders for processed and raw data 
        respectively (e.g., ['processed_folder', 'raw_folder']).
    supervised_data_folders : list of str
        A list containing the names of the supervised data source folders. There should be at least two 
        elements corresponding to different supervised sources.
    remote_data_folder : str
        The folder name for the remote data source.
    list_of_questionnaires : list of str
        A list of questionnaire identifiers to be merged.
    data_format : str
        The file extension or data format (e.g., '.csv').
    output_base : str
        The base directory where the merged questionnaire data will be saved. The output paths are 
        constructed using the provided folder_structure.

    See Also
    --------
    load_task_data : Function to load task data from a CSV file.
    merge_and_save : Function to merge a list of DataFrames and save them to a CSV file.
    """
    
    for task in list_of_questionnaires:

        if task == 'q_IC3_demographics':
            
            # Supervised source 1 (two versions for healthy)
            df_v1 = load_task_data(root, supervised_data_folders[0], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='Healthy_questionnaire')
            df_v1_2 = load_task_data(root, supervised_data_folders[0], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
            df_v1_raw = load_task_data(root, supervised_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True)
            df_v1_raw_2 = load_task_data(root, supervised_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True, version_suffix='Healthy')

            # Supervised source 2
            df_v2 = load_task_data(root, supervised_data_folders[1], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
            df_v2_raw = load_task_data(root, supervised_data_folders[1], folder_structure[1].strip("/"), task, data_format, raw=True)
            
            summary_out = os.path.join(output_base, folder_structure[0].strip("/"), f"{task}{data_format}")
            raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
            if remote_data_folder is not None:
                # Remote data source
                df_cog =  load_task_data(root, remote_data_folder, folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='Healthy_questionnaire')
                df_cog_raw =  load_task_data(root, remote_data_folder, folder_structure[1].strip("/"), task, data_format, raw=True, version_suffix='Healthy')
                merge_and_save([df_v1, df_v1_2, df_v2, df_cog], summary_out)
                merge_and_save([df_v1_raw, df_v1_raw_2, df_v2_raw, df_cog_raw], raw_out)
            else:
                merge_and_save([df_v1, df_v1_2, df_v2], summary_out)
                merge_and_save([df_v1_raw, df_v1_raw_2, df_v2_raw], raw_out)
            

        else:
            # For all other questionnaires
            
            # Supervised source 1
            df_v1 = load_task_data(root, supervised_data_folders[0], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
            df_v1_raw = load_task_data(root, supervised_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True)

            # Supervised source 2
            df_v2 = load_task_data(root, supervised_data_folders[1], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
            df_v2_raw = load_task_data(root, supervised_data_folders[1], folder_structure[1].strip("/"), task, data_format, raw=True)
            
            summary_out = os.path.join(output_base, folder_structure[0].strip("/"), f"{task}{data_format}")
            raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
            if remote_data_folder is not None:
                # Remote data source
                df_cog =  load_task_data(root, remote_data_folder, folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
                df_cog_raw =  load_task_data(root, remote_data_folder, folder_structure[1].strip("/"), task, data_format, raw=True)
                merge_and_save([df_v1, df_v2, df_cog], summary_out)
                merge_and_save([df_v1_raw, df_v2_raw, df_cog_raw], raw_out)
            else:
                merge_and_save([df_v1, df_v2], summary_out)
                merge_and_save([df_v1_raw, df_v2_raw], raw_out)   
            
        # Optionally log progress: print(f'Merged questionnaire data for {task}')



def combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure,
                                         list_of_tasks, df_demographics,
                                         clean_file_extension, data_format, append_data):
    """
    Combine demographic data with cognitive task scores.

    This function iterates over a list of cognitive task files, reads each file,
    cleans the data by dropping duplicates and renaming the summary score column to the task ID,
    and then merges the resulting summary scores into the provided demographics DataFrame.
    Finally, it saves the combined DataFrame as a CSV file.

    Parameters
    ----------
    root_path : str
        The root directory path.
    output_clean_folder : str
        The folder name containing cleaned cognitive data.
    folder_structure : list of str
        List of folder structure components; the first element is used to locate the tasks.
    list_of_tasks : list of str
        List of task file base names (without extension or clean_file_extension).
    df_demographics : pd.DataFrame
        DataFrame containing the demographic data.
    clean_file_extension : str
        Suffix appended to task file names (e.g., '_clean').
    data_format : str
        File format extension (e.g., '.csv').

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing demographics and cognitive task scores.

    Examples
    --------
    >>> merged_df = combine_demographics_and_cognition(
    ...     "/data", "cleaned", ["/tasks"], ["task1", "task2"], demographics_df,
    ...     "_clean", ".csv"
    ... )
    >>> merged_df.head()
    """
    # Build the folder path without changing the working directory
    tasks_folder = os.path.join(root_path, output_clean_folder.strip('/'), folder_structure[0].strip('/'))
    
    # Loop through each cognitive task file
    for task_file in list_of_tasks:
        task_file_path = os.path.join(tasks_folder, f"{task_file}{clean_file_extension}{data_format}")
        try:
            temp_cog = pd.read_csv(task_file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading file {task_file_path}: {e}")
            continue
        
        # Drop duplicate entries for each user and reset the index
        temp_cog = temp_cog.drop_duplicates(subset='user_id', keep='last').reset_index(drop=True)
        
        # Ensure the 'taskID' column exists and use its first value as the new column name
        if 'taskID' not in temp_cog.columns:
            print(f"'taskID' column missing in {task_file_path}. Skipping this file.")
            continue
        
        task_id = temp_cog.loc[0, 'taskID']
        
        # Select only the 'user_id' and 'SummaryScore' columns and rename 'SummaryScore' to the task_id
        temp_cog = temp_cog[['user_id', 'SummaryScore']].rename(columns={'SummaryScore': task_id})
        
        # Merge the cognitive data with the demographics data on 'user_id'
        df_demographics = pd.merge(df_demographics, temp_cog, on='user_id', how='left')
    
    # Save the merged DataFrame
    output_file = os.path.join(tasks_folder, "summary_cognition_and_demographics_new.csv")
    
    if append_data:
        if os.path.exists(output_file):  
            df_append = pd.read_csv(output_file) 
            df_demographics = pd.concat([df_demographics,df_append])
            df_demographics = df_demographics.drop_duplicates(subset='user_id',keep='last').reset_index(drop=True)
            df_demographics.to_csv(output_file, index=False)
        else:
            print('Appending demographics failed because there is no file to append to. The file has been saved without appending.')
            df_demographics.to_csv(output_file, index=False)
    else:
        df_demographics.to_csv(output_file, index=False)
        
    return df_demographics
 

def demographics_preproc(root_path, merged_data_folder, output_clean_folder, questionnaire_name,
                         inclusion_criteria, folder_structure, data_format,
                         clean_file_extension, append_data):
    """
    Preprocess and clean demographic questionnaire data.

    This function loads demographic data from two CSV files (a summary file and a raw file),
    filters the data based on the provided inclusion criteria, cleans and transforms the
    demographic variables, and returns a one-hot encoded DataFrame of the cleaned data.
    The processed DataFrame is also saved to a CSV file.

    Parameters
    ----------
    root_path : str
        Root directory path.
    merged_data_folder : str
        Folder name where the merged data is stored.
    questionnaire_name : str
        Name of the questionnaire (used in file naming).
    inclusion_criteria : list
        List of user IDs to be included in the analysis.
    folder_structure : list of str
        Two-element list with folder paths: first for the summary file, second for the raw file.
    data_format : str
        File extension (e.g. ".csv").
    clean_file_extension : str
        Suffix to be appended to the output file name (e.g. "_clean").

    Returns
    -------
    pd.DataFrame or None
        A one-hot encoded DataFrame containing the cleaned demographic data, or
        None if there was an error loading the data.

    Examples
    --------
    >>> df = demographics_preproc("/data", "merged", "survey", [1, 2, 3],
    ...                           ["/summary", "/raw"], ".csv", "_clean")
    >>> df.head()
    """
    
    # Set working directory
    os.chdir(os.path.join(root_path, merged_data_folder.strip('/')))

    # Construct file paths (strip / if any)
    raw_path = os.path.join(folder_structure[1].strip('/'), f"{questionnaire_name}_raw{data_format}")
    summary_path = os.path.join(folder_structure[0].strip('/'), f"{questionnaire_name}{data_format}")

    try:
        df_dem = pd.read_csv(raw_path, low_memory=False)
        df_dem = df_dem[df_dem.user_id.isin(inclusion_criteria)]
        
        df_dem_summary = pd.read_csv(summary_path, low_memory=False)
        df_dem_summary = df_dem_summary[df_dem_summary.user_id.isin(inclusion_criteria)]
    except Exception as e:
        print(f"Error loading {questionnaire_name}: {e}. File might not exist.")
        return None

    # Drop unwanted columns and duplicates
    if 'Unnamed: 0' in df_dem.columns:
        df_dem.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_dem = df_dem.drop_duplicates(subset=['user_id', 'question'],keep='last').reset_index(drop=True)
    df_dem_summary = df_dem_summary.drop_duplicates(subset='user_id', keep='last').reset_index(drop=True)

    def extract_response(df, question_text):
        """Extract the response and user_id for a given question."""
        return df.loc[df['question'] == question_text, ['user_id', 'response']].copy().reset_index(drop=True)

    # Extract demographics of interest using the question text
    age_df = extract_response(df_dem, '<center>Howoldareyou?</center>')
    gender_df = extract_response(df_dem, '<center>Whatisyourbiologicalsex?</center>')
    education_df = extract_response(df_dem, '<center>Whatisyourhighestlevelofeducation?</center>')
    device_df = extract_response(df_dem, '<center>Whatdeviceareyouusingatthemoment?</center>')
    english_df = extract_response(df_dem, '<center>HowwouldyourateyourproficiencyinEnglish?</center>')
    depression_df = extract_response(df_dem, '<center>Areyoucurrentlytakinganymedicationfordepression?</center>')
    anxiety_df = extract_response(df_dem, '<center>Areyoucurrentlytakinganymedicationforanxiety?</center>')
    dyslexia_df = extract_response(df_dem, '<center>DoyouhaveDyslexia,oranyotherproblemswithreadingandwriting?</center>')
    risks_df = extract_response(df_dem, '<center>Haveyoueverbeentoldyouhavethefollowing?Tickallthatapplies</center>')

    # --- Cleaning Functions for Each Variable ---

    def clean_age(df):
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
        df.loc[df['response'] < 40, 'response'] = np.nan
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'age'}, inplace=True)
        return df

    def clean_gender(df):
        df['response'] = df['response'].replace(['53', '55', '65', '78', '71', '72'], np.nan)
        df.replace({'Male': 0, 'Female': 1}, inplace=True)
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'gender'}, inplace=True)
        return df

    def clean_education(df):
        df['response'] = df['response'].replace('1', np.nan)
        df['response'] = df['response'].replace({
            'Secondary/HighSchoolDiploma': 'Secondary/HighSchool-A-levels',
            'Primary/ElementarySchool': 'SecondarySchool-GCSE'
        })
        mapping = {
            'SecondarySchool-GCSE': 0,
            'Secondary/HighSchool-A-levels': 1,
            'ProfessionalDegree': 1,
            "Bachelor'sDegree": 2,
            "Master'sDegree": 3,
            'PhD': 3
        }
        df.replace({'response': mapping}, inplace=True)
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'education'}, inplace=True)
        return df

    def clean_device(df, summary_df):
        df = df.merge(summary_df[['user_id', 'os']], on='user_id', how='outer')
        df['os'] = df.os.apply(lambda x: ast.literal_eval(x)[0] if ',' in x else x)
        df['response'] = df['response'].fillna(df['os'])
        df['response'] = df['response'].replace({'Mac OS X': 'Tablet', 'Android': 'Phone', 'Windows': 'Laptop/Computer', 'iOS': 'Phone', 'Chrome OS': 'Laptop/Computer'})
        df.drop(columns='os', inplace=True)
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'device'}, inplace=True)
        return df

    def clean_english(df):
        df['response'] = df['response'].replace({'3': 1, '4': 1, 'No': np.nan, '2': 1, '1': 0})
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'english'}, inplace=True)
        return df

    def clean_binary_response(df, col_name, mapping):
        df['response'] = df['response'].replace(mapping)
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': col_name}, inplace=True)
        return df

    def clean_risks(df):
        df.drop_duplicates(keep='last', inplace=True)
        df.replace(np.nan, ' ', inplace=True)
        df['response'] = df['response'].str.lower()
        risk_conditions = {
            'diabetes': ['diabetes'],
            'highbloodpressure': ['highbloodpressure'],
            'highcholesterol': ['highcholesterol', 'highcholesterole'],
            'heartdisease': ['heartdisease'],
            'kidneydisease': ['kidneydisease'],
            'alcoholdependency': ['alcoholdependency'],
            'over-weight': ['over-weight', 'overweight'],
            'long-termsmoker': ['long-termsmoker', 'longtermsmoker'],
            'ex-smoker': ['ex-smoker', 'exsmoker']
        }
        for key, terms in risk_conditions.items():
            df[key] = df['response'].apply(lambda x: any(term in x for term in terms)).astype(int)
        df.loc[(df['long-termsmoker'] & df['ex-smoker']).astype(bool), 'ex-smoker'] = 0
        
        # Sum risk flags to obtain a risk score
        df['response'] = df[list(risk_conditions.keys())].sum(axis=1)
        #df.drop(columns=list(risk_conditions.keys()), inplace=True)
        df.drop_duplicates(subset='user_id', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'response': 'risks'}, inplace=True)
        return df

    # --- Clean each extracted DataFrame ---
    age_df = clean_age(age_df)
    gender_df = clean_gender(gender_df)
    education_df = clean_education(education_df)
    device_df = clean_device(device_df, df_dem_summary)
    english_df = clean_english(english_df)
    depression_df = clean_binary_response(depression_df, 'depression',
                                          {'No': 0, 'Yes': 1, 'SKIPPED': 0})
    anxiety_df = clean_binary_response(anxiety_df, 'anxiety',
                                       {'No': 0, 'Yes': 1, 'SKIPPED': 0})
    dyslexia_df = clean_binary_response(dyslexia_df, 'dyslexia',
                                     {'Yes': 1, 'No': 0, 'Tablet': 0, 'Touchscreen': 0, np.nan: 0})
    risks_df = clean_risks(risks_df)

    # Merge all cleaned data on 'user_id'
    dfs = [age_df, gender_df, education_df, device_df, english_df,
           depression_df, anxiety_df, risks_df, dyslexia_df]
    healthy_demographics = dfs[0]
    
    for df in dfs[1:]:
        healthy_demographics = healthy_demographics.merge(df, how='left', on='user_id')
        
    healthy_demographics['dyslexia'] = healthy_demographics['dyslexia'].replace(np.nan,0)
    healthy_demographics = healthy_demographics.dropna()

    # Ensure education is of integer type
    healthy_demographics['education'] = healthy_demographics['education'].astype(int)

    # One-hot encode categorical variables
    one_hot_encoded_data = pd.get_dummies(healthy_demographics, columns=['device', 'education'])
    one_hot_encoded_data.rename(columns={
        'education_1': 'education_Alevels',
        'education_2': 'education_bachelors',
        'education_3': 'education_postBachelors',
        'device_1': 'device_tablet',
        'device_0': 'device_phone',
        'english': 'english_secondLanguage'
    }, inplace=True)
    one_hot_encoded_data.replace({True: 1, False: 0}, inplace=True)

    # Adjust numerical columns by subtracting 0.5 (from gender to education_postBachelors)
    cols_to_adjust = one_hot_encoded_data.loc[:, 'gender':'education_postBachelors'].columns
    one_hot_encoded_data[cols_to_adjust] = one_hot_encoded_data[cols_to_adjust] - 0.5

    # Save the final DataFrame
    output_path = os.path.join(root_path, output_clean_folder.strip('/'), folder_structure[0].strip('/'),
                               f"{questionnaire_name}{clean_file_extension}{data_format}")
    
    if append_data:
        if os.path.exists(output_path):  
            df_append = pd.read_csv(output_path) 
            one_hot_encoded_data = pd.concat([one_hot_encoded_data,df_append])
            one_hot_encoded_data = one_hot_encoded_data.drop_duplicates(subset='user_id',keep='last').reset_index(drop=True)
            one_hot_encoded_data.to_csv(output_path, index=False)
        else:
            print('Appending demographics failed because there is no file to append to. The file has been saved without appending.')
            one_hot_encoded_data.to_csv(output_path, index=False)
    else:
        one_hot_encoded_data.to_csv(output_path, index=False)
        
    return one_hot_encoded_data

  
def remove_general_outliers(root_path, merged_data_folder, task_name, inclusion_criteria,  data_format, folder_structure=['/summary_data','/trial_data','/speech']):
    
    """
    Remove general outliers from merged data based on inclusion criteria.

    This function loads processed and raw data for a given task from designated directories,
    filters the data to retain only rows with user IDs present in the inclusion criteria, and 
    cleans the data by removing duplicates and unnecessary columns. If the files cannot be 
    loaded, an error message is printed and the function returns (None, None).

    Parameters
    ----------
    root_path : str
        The base directory path where the data is stored.
    merged_data_folder : str
        The folder name or relative path containing the merged data; concatenated with root_path to form the full data path.
    task_name : str
        The name of the task corresponding to the data files to be loaded.
    inclusion_criteria : array_like
        A collection of user identifiers. Only rows with a user_id that is in this collection will be kept.
    data_format : str
        The file extension for the data files (e.g., '.csv').
    folder_structure : list of str, optional
        A list of subfolder names used to locate the summary and raw data files. The first element is used 
        for processed/summary data, and the second element for trial/raw data. Default is ['/summary_data', '/trial_data', '/speech'].

    Returns
    -------
    tuple of (pandas.DataFrame or None, pandas.DataFrame or None)
        A tuple containing two DataFrames:
        - The first DataFrame is the cleaned summary data with duplicates removed and specified columns dropped.
        - The second DataFrame is the cleaned raw data with duplicates removed; if a column named 'Unnamed: 0'
          is present, it is renamed to 'Level_filter'.
        If an error occurs during file loading (e.g., if the file does not exist), the function prints an error 
        message and returns (None, None).

    Examples
    --------
    >>> df_summary, df_raw = remove_general_outliers("/data/", "merged/", "task1", [1, 2, 3], ".csv")
    >>> if df_summary is not None:
    ...     print(df_summary.head())
    """
        
    path_to_data = root_path + merged_data_folder
    
    try:
        df = pd.read_csv(f'{path_to_data}/{folder_structure[0]}/{task_name}{data_format}', low_memory=False)
        df = df[df.user_id.isin(inclusion_criteria)]
        
        df_raw = pd.read_csv(f'{path_to_data}/{folder_structure[1]}/{task_name}_raw{data_format}', low_memory=False)
        df_raw = df_raw[df_raw.user_id.isin(inclusion_criteria)]
    except:
        print(f'Error in loading {task_name}. File might not exist.')
        return None,None

    df.drop_duplicates(subset=['user_id'],keep="last", inplace=True)
    df.drop(columns=['Unnamed: 0','Level','type','RespObject','sequenceObj', 'dynamicDifficulty'], inplace=True)
    df.reset_index(drop=True,inplace=True)

    if ('Unnamed: 0' in df_raw.columns):
        
        df_raw = df_raw.rename(columns={'Unnamed: 0':'Level_filter'})

    df_raw.drop_duplicates(subset=['user_id','Level_filter'],keep="last", inplace=True)
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw


def output_preprocessed_data(df,df_raw, root_path, output_clean_folder, folder_structure, clean_file_extension, data_format, append_data):
    
    """
    Output pre-processed data to designated clean folders.

    This function changes the current working directory to the specified root path, checks for the
    existence of the output folder and its required subdirectories (as defined in folder_structure),
    and creates them if they do not exist. It then writes both the processed and raw DataFrames to CSV
    files. The filenames are constructed using the 'taskID' from the first row of the processed DataFrame,
    appending the provided clean_file_extension and data_format.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed DataFrame to be saved.
    df_raw : pandas.DataFrame
        The raw preprocessed DataFrame to be saved.
    root_path : str
        The root directory path where the output directories should be created.
    output_clean_folder : str
        The folder path for the cleaned output data. The leading character (if present) is removed when
        checking directory existence.
    folder_structure : list of str
        A list of subfolder names that define where to save the processed and raw data respectively.
        For example, ['/processed', '/raw'].
    clean_file_extension : str
        A suffix to append to the filename (e.g., '_clean') before the data format extension.
    data_format : str
        The file extension indicating the data format (e.g., '.csv').

    """
    
    os.chdir(root_path)
    
    if os.path.isdir(output_clean_folder[1:]) == False:
        os.mkdir(output_clean_folder[1:])
        os.mkdir(f'.{output_clean_folder}{folder_structure[0]}')
        os.mkdir(f'.{output_clean_folder}{folder_structure[1]}')
        
    if append_data:
        df_temp = pd.read_csv(f".{output_clean_folder}{folder_structure[0]}/{df.loc[0,'taskID']}{clean_file_extension}{data_format}")
        df_temp_raw = pd.read_csv(f".{output_clean_folder}{folder_structure[1]}/{df.loc[0,'taskID']}_raw{clean_file_extension}{data_format}")
        df = pd.concat([df_temp,df], axis=0).drop_duplicates(subset='user_id', keep='last').reset_index(drop=True)
        df_raw = pd.concat([df_temp_raw,df_raw], axis=0).drop_duplicates(subset='user_id', keep='last').reset_index(drop=True)  

    df.to_csv(f".{output_clean_folder}{folder_structure[0]}/{df.loc[0,'taskID']}{clean_file_extension}{data_format}", index=False)
    df_raw.to_csv(f".{output_clean_folder}{folder_structure[1]}/{df.loc[0,'taskID']}_raw{clean_file_extension}{data_format}")

    return None

def orientation_preproc(df,df_raw):
    
    """
    Pre-process orientation task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Pear Cancellation task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Digit Span task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Spatial Span task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Paired Associates Learning task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Semantic Judgement task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
        for x in range (len(temp_score)):
            if (temp_score[x] >3) & (x!=5):
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
    
    """
    Pre-process SRT task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
    df_raw.loc[df_raw.RT<180,'RT'] = np.nan
    df_raw.loc[df_raw.RT>800,'RT'] = np.nan

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

    exc = ((df.timeOffScreen > 10000) | (df.focusLossCount > 2) | (df.numCorrectClicks < 30) | (df.numTimeOuts > 14) | (df.numMisclicks > 14))    
    df.drop(df[exc].index, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def crt_preproc(df,df_raw):
    
    """
    Pre-process CRT task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Motor control task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Task Recall task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Auditory Attention task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Calculation task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Blocks task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Gesture Recognition task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Rule Learning task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Odd One Out task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Language Comprehension task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
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
    
    """
    Pre-process Trail-making task data and compute performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing participant information. 
    df_raw : pandas.DataFrame
        Raw DataFrame for the task.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - df : The updated summary DataFrame with computed performance metrics
        - df_raw : The filtered raw DataFrame that includes only those users present in the
          updated summary DataFrame.
    """
    
    df_raw.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw.loc[df_raw.RT>10000,'RT'] = np.nan
    
    df_raw2.loc[df_raw2.RT<200,'RT'] = np.nan
    df_raw2.loc[df_raw2.RT>10000,'RT'] = np.nan

    df_raw3.loc[df_raw3.RT<200,'RT'] = np.nan
    df_raw3.loc[df_raw3.RT>10000,'RT'] = np.nan

    scores = [None] * len(df.user_id)
    errors = [None] * len(df.user_id)
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