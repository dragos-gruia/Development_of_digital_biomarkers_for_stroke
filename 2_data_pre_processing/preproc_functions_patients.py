"""
Last updated on 25th of March 2025
@authors: Dragos Gruia
"""

import os
import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings('ignore')

def main_preprocessing(root_path, list_of_tasks, list_of_questionnaires, list_of_speech, 
                       clinical_information=None, 
                       patient_data_folders=['/data_patients_v1', '/data_patients_v2'],
                       folder_structure=['/summary_data', '/trial_data', '/speech'], 
                       output_clean_folder='/data_patients_cleaned', 
                       merged_data_folder='/data_patients_combined', 
                       clean_file_extension='_cleaned', 
                       data_format='.csv',
                       append_data = False):
    """
    Perform the main preprocessing workflow for patient data.

    This function merges patient data across sites, preprocesses cognitive task and questionnaire 
    data, and finally combines the demographics with cognitive scores (and optional clinical 
    information) into a single summary file. The preprocessing steps include:
    
    1. Merging data across sites using the patient data folders.
    2. For each cognitive task provided in `list_of_tasks`, loading the corresponding merged data,
       removing general outliers, applying a task-specific preprocessing function (using a match-case
       construct), and saving the preprocessed output.
    3. For each questionnaire provided in `list_of_questionnaires`, applying a questionnaire-specific 
       preprocessing function.
    4. Combining the preprocessed demographics with the cognitive task scores. If clinical 
       information is provided, the clinical data is merged and additional date and duration 
       calculations are performed.
    5. Saving the final combined DataFrame as an Excel file.

    Parameters
    ----------
    root_path : str
        The root directory where the data is stored.
    list_of_tasks : list of str
        A list of cognitive task names to preprocess.
    list_of_questionnaires : list of str
        A list of questionnaire names to preprocess.
    list_of_speech : list of str
        A list of speech task names to merge.
    clinical_information : str, optional
        Path to an Excel file containing clinical information. If None, clinical data is not merged.
    patient_data_folders : list of str, optional
        List of folder names for patient data sources (default is ['/data_patients_v1', '/data_patients_v2']).
    folder_structure : list of str, optional
        List of folder names for summary, trial, and speech data (default is ['/summary_data', '/trial_data', '/speech']).
    output_clean_folder : str, optional
        The folder where the cleaned data will be saved (default is '/data_patients_cleaned').
    merged_data_folder : str, optional
        The folder where merged data is stored (default is '/data_patients_combined').
    clean_file_extension : str, optional
        Suffix appended to preprocessed file names (default is '_cleaned').
    data_format : str, optional
        File extension (default is '.csv').
    append_data : bool, optional
        Append data to existing file or overwrite existing excel file (default is False).

    Returns
    -------
    pandas.DataFrame
        The final combined DataFrame containing demographics, cognitive scores, and, if available,
        clinical information. This DataFrame is also saved as an Excel file named
        'summary_cognition_and_demographics.xlsx'.

    Examples
    --------
    >>> combined_df = main_preprocessing(
    ...     "/data", 
    ...     list_of_tasks=["IC3_Orientation", "IC3_TaskRecall"],
    ...     list_of_questionnaires=["q_IC3_demographics", "q_IC3_IADL"],
    ...     list_of_speech=["SpeechTask1"],
    ...     clinical_information="clinical_info.xlsx"
    ... )
    >>> combined_df.head()
    """
    
    print('Starting preprocessing...')
    
    # Change working directory to root_path.
    os.chdir(root_path)

    # Initialize empty DataFrames.
    df_demographics = pd.DataFrame()
    df_iadl = pd.DataFrame()
    df_combined = pd.DataFrame()

    # Merge patient data across sites.
    if len(patient_data_folders) > 1:
        print('Merging data across sites...', end="", flush=True)
        merge_patient_data_across_sites(
            root_path, folder_structure, patient_data_folders, 
            list_of_tasks, list_of_questionnaires, list_of_speech, 
            data_format, merged_data_folder
        )
        print('Done')
    else:
        merged_data_folder = patient_data_folders[0]

    # Process cognitive tasks.
    if list_of_tasks is not None:
        for task_name in list_of_tasks:
            print(f'Pre-processing {task_name}...', end="", flush=True)
            df, df_raw = remove_general_outliers(root_path, merged_data_folder, task_name, data_format, folder_structure)
            
            # Use match-case to apply task-specific preprocessing.
            match task_name:
                case 'IC3_Orientation':
                    df, df_raw = orientation_preproc(df, df_raw)
                case 'IC3_TaskRecall':
                    df, df_raw = taskrecall_preproc(df, df_raw)
                case 'IC3_rs_PAL':
                    df, df_raw = pal_preproc(df, df_raw)
                case 'IC3_rs_digitSpan':
                    df, df_raw = digitspan_preproc(df, df_raw)
                case 'IC3_rs_spatialSpan':
                    df, df_raw = spatialspan_preproc(df, df_raw)
                case 'IC3_Comprehension':
                    df, df_raw = comprehension_preproc(df, df_raw)
                case 'IC3_SemanticJudgment':
                    df, df_raw = semantics_preproc(df, df_raw)
                case 'IC3_BBCrs_blocks':
                    df, df_raw = blocks_preproc(df, df_raw)
                case 'IC3_NVtrailMaking':
                    df2, df_raw2 = remove_general_outliers(root_path, merged_data_folder, f'{task_name}2', data_format, folder_structure)
                    df3, df_raw3 = remove_general_outliers(root_path, merged_data_folder, f'{task_name}3', data_format, folder_structure)
                    df, df_raw = trailmaking_preproc(df, df2, df3, df_raw, df_raw2, df_raw3, task_name)
                case 'IC3_rs_oddOneOut':
                    df, df_raw = oddoneout_preproc(df, df_raw)
                case 'IC3_i4i_IDED':
                    df, df_raw = rule_learning_preproc(df, df_raw)
                case 'IC3_PearCancellation':
                    df, df_raw = pear_cancellation_preproc(df, df_raw)
                case 'IC3_rs_SRT':
                    df, df_raw = srt_preproc(df, df_raw)
                case 'IC3_AuditorySustainedAttention':
                    df, df_raw = auditory_attention_preproc(df, df_raw)
                case 'IC3_rs_CRT':
                    df, df_raw = crt_preproc(df, df_raw)
                case 'IC3_i4i_motorControl':
                    df, df_raw = motor_control_preproc(df, df_raw)
                case 'IC3_calculation':
                    df, df_raw = calculation_preproc(df, df_raw)
                case 'IC3_GestureRecognition':
                    df, df_raw = gesture_preproc(df, df_raw)
                case _:
                    print(f'Task {task_name} does not have a specific preprocessing function.')
                    continue
            
            # Save the preprocessed task output.
            output_preprocessed_data(df, df_raw, root_path, output_clean_folder, folder_structure, clean_file_extension, data_format, append_data)
            print('Done')
    else:
        print('No tasks were provided.')
    
    # Process questionnaires.
    harcode_cleaning  = False if len(patient_data_folders) == 1 else True
    if list_of_questionnaires is not None:
        for questionnaire_name in list_of_questionnaires:
            print(f'Pre-processing {questionnaire_name}...', end="", flush=True)
            
            match questionnaire_name:
                case 'q_IC3_demographics':
                    df_demographics = demographics_preproc(
                        root_path, merged_data_folder, output_clean_folder,
                        questionnaire_name, folder_structure, data_format, clean_file_extension, harcode_cleaning
                    )
                case 'q_IC3_IADL':
                    df_iadl = iadl_preproc(
                        root_path, merged_data_folder, questionnaire_name,
                        folder_structure, data_format, clean_file_extension, output_clean_folder
                    )
                case _:
                    print(f'Questionnaire {questionnaire_name} does not have a specific preprocessing function.')
            print('Done')
    else:
        print('No questionnaires were provided.')
    
    # Combine demographics with cognitive data.
    if not df_demographics.empty:
        df_combined = combine_demographics_and_cognition(
            root_path, output_clean_folder, folder_structure, list_of_tasks,
            df_demographics, clean_file_extension, data_format, clinical_information, harcode_cleaning
        )
    
    # Merge IADL summary with combined data if available.
    try:
        df_combined = df_combined.merge(df_iadl, on='user_id', how='left')
    except:
        print('IADL does not exist and it was now merged.')
        
    print('Preprocessing complete.')
    
    # Save final combined DataFrame as an Excel file.
    output_excel = os.path.join(root_path, output_clean_folder.strip('/'), folder_structure[0].strip('/'), "summary_cognition_and_demographics.xlsx")
    
    if append_data:
        if os.path.exists(output_excel):  
            df_append = pd.read_excel(output_excel) 
            df_combined = pd.concat([df_combined,df_append])
            df_combined = df_combined.drop_duplicates(subset='user_id',keep='last').reset_index(drop=True)
            df_combined.to_excel(output_excel, index=False)
            print('Data appended successfully.') 
        else:
            print('Appending failed because there is no file to append to. The file has been saved without appending.')
            df_combined.to_excel(output_excel, index=False)
    else:
        df_combined.to_excel(output_excel, index=False)
    
    return df_combined


def ensure_directory(path):
    """Checks if directory exists, creates it if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def merge_patient_data_across_sites(root_path, folder_structure, patient_data_folders, list_of_tasks, list_of_questionnaires,
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
    patient_data_folders : list of str
        A list of folder names containing patient data.
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
    merge_clinical_tests(root_path, folder_structure, patient_data_folders, list_of_tasks, data_format, output_base)
    
    # Merge speech data
    merge_speech_data(root_path, folder_structure, patient_data_folders, list_of_speech,  data_format, output_base)
    
    # Merge questionnaire data
    merge_questionnaire_data(root_path, folder_structure, patient_data_folders,
                             list_of_questionnaires,  data_format, output_base)
    
    return None


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

def merge_clinical_tests(root, folder_structure, patient_data_folders, tasks, data_format, output_base):
    
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
    patient_data_folders : list of str
        A list containing the names of the patient data source folders.
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
        df_v1 = load_task_data(root, patient_data_folders[0], folder_structure[0], task, data_format, raw=False)
        df_v1_raw = load_task_data(root, patient_data_folders[0], folder_structure[1], task, data_format, raw=True)
        
        # Supervised source 2 (with special case for IDED)
        if task == "IC3_i4i_IDED":
            df_v2 = load_task_data(root, patient_data_folders[1], folder_structure[0], task, data_format, raw=False, version_suffix='2')
            df_v2_raw = load_task_data(root, patient_data_folders[1], folder_structure[1], task, data_format, raw=True, version_suffix='2')
        else:
            df_v2 = load_task_data(root, patient_data_folders[1], folder_structure[0], task, data_format, raw=False)
            df_v2_raw = load_task_data(root, patient_data_folders[1], folder_structure[1], task, data_format, raw=True)
                
        # Merge the three sources
        summary_out = os.path.join(output_base, folder_structure[0].strip("/"), f"{task}{data_format}")
        merge_and_save([df_v1, df_v2], summary_out)
        
        raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
        merge_and_save([df_v1_raw, df_v2_raw],raw_out)

        # Optionally log progress: print(f'Merged clinical test data for {task}')

def merge_speech_data(root, folder_structure, patient_data_folders, list_of_speech,  data_format, output_base):
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
    patient_data_folders : list of str
        A list containing the names of the patient data folders. There should be at least two 
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
        df_v1_raw = load_task_data(root, patient_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True)
        df_v2_raw = load_task_data(root, patient_data_folders[1], folder_structure[1].strip("/"), task, data_format, raw=True)
        
        raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
        merge_and_save([df_v1_raw, df_v2_raw], raw_out)
        
        
        # Optionally log progress: print(f'Merged speech data for {task}')

def merge_questionnaire_data(root, folder_structure, patient_data_folders, list_of_questionnaires,  data_format, output_base):
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
    patient_data_folders : list of str
        A list containing the names of the patient data source folders. There should be at least two 
        elements corresponding to different supervised sources.
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
        
        # For all other questionnaires
        # Supervised source 1

        df_v1 = load_task_data(root, patient_data_folders[0], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
        df_v1_raw = load_task_data(root, patient_data_folders[0], folder_structure[1].strip("/"), task, data_format, raw=True)

        # Supervised source 2
        df_v2 = load_task_data(root, patient_data_folders[1], folder_structure[0].strip("/"), task, data_format, raw=False, version_suffix='_questionnaire')
        df_v2_raw = load_task_data(root, patient_data_folders[1], folder_structure[1].strip("/"), task, data_format, raw=True)
        

        summary_out = os.path.join(output_base, folder_structure[0].strip("/"), f"{task}{data_format}")
        merge_and_save([df_v1, df_v2], summary_out)
        
        raw_out = os.path.join(output_base, folder_structure[1].strip("/"), f"{task}_raw{data_format}")
        merge_and_save([df_v1_raw, df_v2_raw], raw_out)
        
        # Optionally log progress: print(f'Merged questionnaire data for {task}')

import os
import pandas as pd
import numpy as np
import datetime


def combine_demographics_and_cognition(root_path, output_clean_folder, folder_structure,
                                       list_of_tasks, df_demographics, clean_file_extension,
                                       data_format, clinical_information, harcode_cleaning):
    """
    Combine demographics data with cognitive task scores and optional clinical information.

    This function reads cognitive task files for healthy participants, merges the corresponding
    summary scores into a demographics DataFrame, and then links participant identifiers and
    timepoints based on their user IDs. Optionally, clinical information is merged into the
    combined DataFrame. The final merged data is saved to an Excel file.

    Parameters
    ----------
    root_path : str
        The root directory path.
    output_clean_folder : str
        The folder name containing the cleaned cognitive data.
    folder_structure : list of str
        A list of folder paths. The first element is used to locate task files.
    list_of_tasks : list of str
        List of task file base names (without extension or clean_file_extension).
    df_demographics : pandas.DataFrame
        DataFrame containing demographic data. Must include the column 'user_id'.
    clean_file_extension : str
        Suffix appended to task file names (e.g., "_clean").
    data_format : str
        File extension (e.g., ".csv").
    clinical_information : str or None
        Path to an Excel file containing clinical information. If None, clinical data is not merged.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined demographics, cognitive scores, and optional clinical
        information. The result is also saved to an Excel file named "summary_cognition_and_demographics.xlsx".

    Examples
    --------
    >>> combined_df = combine_demographics_and_cognition(
    ...     "/data", "cleaned", ["/tasks", "/raw"],
    ...     ["IC3_Task1", "IC3_Orientation"], demographics_df, "_clean", ".csv", "clinical_info.xlsx"
    ... )
    >>> combined_df.head()
    """
    # Build the folder path for task files without changing the working directory.
    tasks_folder = os.path.join(root_path, output_clean_folder.strip('/'),
                                folder_structure[0].strip('/'))

    # Merge each task's cognitive data with demographics.
    for task_file in list_of_tasks:
        file_path = os.path.join(tasks_folder, f"{task_file}{clean_file_extension}{data_format}")
        try:
            temp_cog = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Remove duplicates and reset index.
        temp_cog.drop_duplicates(subset='user_id', keep='last', inplace=True)
        temp_cog.reset_index(drop=True, inplace=True)

        # Use the first taskID value as the new column name.
        task_id = temp_cog['taskID'].iloc[0]

        # Select columns based on task type.
        if task_file == 'IC3_Orientation':
            temp_cog = temp_cog.loc[:, ['user_id', 'startTime', 'SummaryScore']]
        else:
            temp_cog = temp_cog.loc[:, ['user_id', 'SummaryScore']]

        temp_cog.rename(columns={'SummaryScore': task_id}, inplace=True)
        df_demographics = pd.merge(df_demographics, temp_cog, on="user_id", how="left")

    # Link patients based on ID and timepoint.
    df_demographics['ID'] = np.nan
    df_demographics['timepoint'] = np.nan

    # Ensure user_id is string for processing.
    df_demographics['user_id'] = df_demographics['user_id'].astype(str)
    
    # Assign timepoint based on the presence of session labels.
    df_demographics.loc[df_demographics.user_id.str.contains('session1'), 'timepoint'] = 1
    df_demographics.loc[df_demographics.user_id.str.contains('session2'), 'timepoint'] = 2
    df_demographics.loc[df_demographics.user_id.str.contains('session3'), 'timepoint'] = 3
    df_demographics.loc[df_demographics.user_id.str.contains('session4'), 'timepoint'] = 4

    # Extract the participant ID from the user_id if it has three parts.
    def extract_id(uid):
        parts = uid.split('-')
        if len(parts) == 3:
            base_id = parts[0]
            if 'ic3study' in base_id.lower():
                return base_id[8:]
            return base_id
        return uid

    df_demographics['ID'] = df_demographics['user_id'].apply(extract_id)

    # Correct naming and technical errors using external functions.
    if harcode_cleaning:
        df_demographics = fix_naming_errors(df_demographics)
        df_demographics = remove_technical_errors(df_demographics)

    # Merge clinical information if provided.
    if clinical_information is not None:
        try:
            df_clinical = pd.read_excel(clinical_information, sheet_name='Patient ')
        except Exception as e:
            print(f"Error reading clinical information: {e}")
            return df_demographics

        clinical_cols = ['STUDY ID', 'CVA date', 'CVA aetiology', 'vascular teritory',
                         'lesion site', 'Aphasia', 'CVA non-cognitive deficit',
                         'NIHSS at admission or 2 hours after thrombectomy/thrombolysis']
        df_clinical = df_clinical.loc[:, clinical_cols]
        df_clinical.rename(columns={'STUDY ID': 'ID'}, inplace=True)
        df_clinical['ID'] = df_clinical['ID'].apply(lambda x: x.strip() if pd.notnull(x) else x)

        df_combined = pd.merge(df_demographics, df_clinical, on='ID', how='left')
        df_combined.drop_duplicates(subset='user_id', inplace=True)
        df_combined.dropna(subset=['startTime'], inplace=True)

        # Convert startTime to datetime.
        df_combined['date_of_ic3'] = df_combined['startTime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
        df_combined['date_of_ic3'] = pd.to_datetime(df_combined['date_of_ic3'])
        df_combined['CVA date'] = pd.to_datetime(df_combined['CVA date'])
        df_combined['days_since_stroke'] = (df_combined['date_of_ic3'] - df_combined['CVA date']).dt.days

        output_file = os.path.join(root_path, "summary_cognition_and_demographics.xlsx")
        df_combined.to_excel(output_file, index=False)
        return df_combined
    else:
        output_file = os.path.join(root_path, "summary_cognition_and_demographics.xlsx")
        df_demographics.to_excel(output_file, index=False)
        return df_demographics

def remove_technical_errors(df_demographics):
    """
    Remove technical errors from the demographics DataFrame.

    This function removes known problematic participant records and corrects task scores
    by setting them to NaN for specific participant IDs where technical errors were detected.
    The processing steps include:
      - Dropping participant records with known faulty indices.
      - For specific participant IDs, replacing values in the affected task score columns
        with NaN.
      - Resetting the DataFrame index after modifications.

    Parameters
    ----------
    df_demographics : pandas.DataFrame
        The demographics DataFrame containing participant data. It must include an 'ID'
        column along with task score columns such as 'IC3_PearCancellation',
        'IC3_SemanticJudgment', 'IC3_GestureRecognition', 'IC3_rs_CRT', 'IC3_Comprehension',
        and 'IC3_calculation'.

    Returns
    -------
    pandas.DataFrame
        The updated demographics DataFrame with technical errors removed.

    Examples
    --------
    >>> df_clean = remove_technical_errors(df_demographics)
    >>> df_clean.head()
    """
    
    # Drop known problematic participant records by index.
    indices_to_drop = ['ic3study10039-session1-versionA', 'ic3study00097-session1-versionA']
    try:
        df_demographics.drop(index=indices_to_drop, inplace=True, errors='ignore')
    except Exception as e:
        print(f"User ID could not be found {e}.")

    # Define mapping of participant IDs to the columns that should be set to NaN.
    corrections = {
        'Anon-4401C1A6EEB843FBA061A62EC8C1E47D': ['IC3_PearCancellation', 'IC3_SemanticJudgment'],
        '00008-session1-versionA': ['IC3_GestureRecognition'],
        '00009-session1-versionA': ['IC3_SemanticJudgment'],
        'ic3study00018-session1-versionA': ['IC3_rs_CRT'],
        'ic3study00033-session1-versionA': ['IC3_rs_CRT'],
        'ic3study00041-session1-versionA': ['IC3_GestureRecognition'],
        'ic3study00090-session1-versionA': ['IC3_rs_CRT'],
        'ic3study00095-session1-versionA': ['IC3_Comprehension'],
        'ic3study00124-session1-versionA': ['IC3_calculation']
    }

    # Apply corrections: set specified columns to NaN for matching participant IDs.
    for participant_id, columns in corrections.items():
        mask = df_demographics['user_id'] == participant_id
        for col in columns:
            df_demographics.loc[mask, col] = np.nan

    # Reset the DataFrame index.
    df_demographics.reset_index(drop=True, inplace=True)
    return df_demographics
        
def fix_naming_errors(df_demographics):
    """
    Correct naming errors in the demographics DataFrame.

    This function applies a set of predefined corrections to fix erroneous
    participant identifiers and timepoint values in the demographics DataFrame.
    The corrections are specified as a mapping from incorrect index labels
    to their correct 'ID' and 'timepoint' values. Finally, rows missing the
    'user_id' are dropped.

    Parameters
    ----------
    df_demographics : pandas.DataFrame
        The demographics DataFrame indexed by user identifiers that may contain
        naming errors. Expected to have at least the columns 'ID', 'timepoint',
        and 'user_id'.

    Returns
    -------
    pandas.DataFrame
        The updated demographics DataFrame with corrected participant identifiers
        and timepoints.

    Examples
    --------
    >>> df_fixed = fix_naming_errors(df_demographics)
    >>> df_fixed.loc['ic300005s1o1', ['ID', 'timepoint']]
    ID            00005
    timepoint        1
    Name: ic300005s1o1, dtype: object
    """
    corrections = {
        'ic300005s1o1': {'ID': '00005', 'timepoint': 1},
        '00008-session1-versionA': {'ID': '00008', 'timepoint': 1},
        'Anon-4401C1A6EEB843FBA061A62EC8C1E47D': {'ID': '00007', 'timepoint': 1},
        '00011-session1-versiona': {'ID': '00011', 'timepoint': 1},
        '00012-session1-versionA': {'ID': '00012', 'timepoint': 1},
        '00009': {'ID': '00004', 'timepoint': 1},
        'ic3study00015-session1-versionA': {'ID': '00015', 'timepoint': 1},
        'ic3study00016-session1-versionA': {'ID': '00016', 'timepoint': 1},
        'bbrandon@hotmail.co.uk': {'ID': '00014', 'timepoint': 1},
        'simoko.hart@gmail.com': {'ID': '00017', 'timepoint': 1},
        'ic3study00018-session1-versionA': {'ID': '00018', 'timepoint': 1},
        'ic3study00019-session1-versionA': {'ID': '00019', 'timepoint': 1},
        'ic3study00020-session1-versionA': {'ID': '00020', 'timepoint': 1},
        'ic3study00021-session1-versionA': {'ID': '00021', 'timepoint': 1},
        'ic3study00024-session1-versionA': {'ID': '00024', 'timepoint': 1},
        'ic3study00022-session1-versionA': {'ID': '00022', 'timepoint': 1},
        'ic3study00023-session1-versionA': {'ID': '00023', 'timepoint': 1},
        'ic3study00027-session1-versionA': {'ID': '00027', 'timepoint': 1},
        'ic3study00032-session1-versionA': {'ID': '00032', 'timepoint': 2},
        '00032-session1-versionA': {'ID': '00032', 'timepoint': 3},
        'ic3study00033-session1-versionA': {'ID': '00033', 'timepoint': 1},
        'ic3study00036-session1-versionA': {'ID': '00036', 'timepoint': 1},
        'ic3study00040-session1-versionA': {'ID': '00040', 'timepoint': 1},
        'ic3study00041-session1-versionA': {'ID': '00041', 'timepoint': 1},
        'ic3study00010-session2-versionA': {'ID': '00010', 'timepoint': 2},
        'ic3study00020-session2-versionB': {'ID': '00020', 'timepoint': 2},
        'ic3study00042-session1-versionA': {'ID': '00042', 'timepoint': 1},
        'ic3study00031-session2-versionB': {'ID': '00031', 'timepoint': 2},
        'ic3study00003-session2-versionB': {'ID': '00003', 'timepoint': 2},
        'ic3study20010session2-versionB': {'ID': '20010', 'timepoint': 2},
        'ic3study00098session2-versionB': {'ID': '00098', 'timepoint': 2},
        'ic3study00099session2-versionB': {'ID': '00099', 'timepoint': 2},
        'ic3study00070session3-versionA': {'ID': '00070', 'timepoint': 3},
        'ic3study0110-session2-versionB': {'ID': '00110', 'timepoint': 2},
        'ic3study00014-session1-versionA': {'ID': '00014', 'timepoint': 2},
        'ic3study00004-session2-versionA': {'ID': '00004', 'timepoint': 2},
        'ic3study00003-session3': {'ID': '00003', 'timepoint': 3},
        'ic3study00096session2-versionB': {'ID': '00096', 'timepoint': 2},
        'ic3study000865-session1-versionA': {'ID': '00086'},
        'Anon-3E31422CD53341458C2EAA17599A1BEF': {'ID': '00064', 'timepoint': 1},
        'Anon-0CFAD094D29E484AA02DCE99FACECBC3': {'ID': '00097', 'timepoint': 1},
        'Anon-E3D074ECDB7E4856B7968501981E76BC': {'ID': '00079', 'timepoint': 2},
        'Anon-CAB8613480C04EBB8BE3BFFBE310558F': {'ID': '00042', 'timepoint': 4},
        'Anon-1BD5269BC73644E3907617937B23B7C1': {'ID': '00043', 'timepoint': 4},
        'Anon-F98995DC24CD48EAAA0945FCCE70FA86': {'ID': '00018', 'timepoint': 4},
    }

    for idx, updates in corrections.items():
        for col, new_value in updates.items():
            df_demographics.loc[idx, col] = new_value

    # Drop rows with missing 'user_id'
    df_demographics.dropna(subset=['user_id'], inplace=True)
    return df_demographics

def iadl_preproc(root_path, merged_data_folder, questionnaire_name, folder_structure,
                 data_format, clean_file_extension, output_clean_folder):
    """
    Preprocess the Instrumental Activities of Daily Living (IADL) questionnaire data.

    This function reads raw IADL questionnaire data, cleans and transforms the responses,
    computes a summary IADL score per participant, and saves the summary as a CSV file.
    The processing steps include:
      - Loading the raw data file from the specified folder.
      - Dropping extraneous columns, duplicate rows (based on 'question' and 'user_id'),
        and rows with missing responses.
      - Extracting a numeric score from the first character of each response.
      - Applying custom transformation functions for specific questionnaire items.
      - Summing the transformed scores per participant to obtain the overall IADL score.
      - Saving the resulting summary DataFrame to a designated output folder.

    Parameters
    ----------
    root_path : str
        The root directory path.
    merged_data_folder : str
        The folder where merged data is stored.
    questionnaire_name : str
        The name of the questionnaire (used for file naming).
    folder_structure : list of str
        A list of folder paths. The second element is used for the raw data file.
    data_format : str
        The file extension (e.g., ".csv").
    clean_file_extension : str
        Suffix to be appended to the output file name (e.g., "_clean").
    output_clean_folder : str
        The folder where the processed output file should be saved.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the summary IADL scores for each participant,
        or None if an error occurred during data loading.

    Examples
    --------
    >>> summary_df = iadl_preproc(
    ...     "/data", "merged_data", "iadl", ["/summary", "/raw"],
    ...     ".csv", "_clean", "output"
    ... )
    >>> summary_df.head()
    """
    # Construct the raw data file path
    raw_path = os.path.join(root_path, merged_data_folder.strip('/'),
                            folder_structure[1].strip('/'),
                            f"{questionnaire_name}_raw{data_format}")
    try:
        df_iadl = pd.read_csv(raw_path, low_memory=False)
    except Exception as e:
        print(f"Error in loading {questionnaire_name}: {e}. File might not exist.")
        return None

    # Clean the raw DataFrame
    if 'Unnamed: 0' in df_iadl.columns:
        df_iadl.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_iadl.drop_duplicates(subset=['question', 'user_id'], keep='last', inplace=True)
    df_iadl.dropna(subset=['response'], inplace=True)
    df_iadl.reset_index(drop=True, inplace=True)

    # Extract the initial numeric score from the response string
    df_iadl['score'] = df_iadl['response'].str[0].astype(int)

    # Define transformation functions for specific questionnaire items
    transformations = {
        '<center>Abilitytousetelephone</center>': lambda x: 1 if x != 4 else 0,
        '<center>Shopping</center>': lambda x: 1 if x == 1 else 0,
        '<center>Foodpreparation</center>': lambda x: 1 if x == 1 else 0,
        '<center>Housekeeping</center>': lambda x: 1 if x != 5 else 0,
        '<center>Laundry</center>': lambda x: 1 if x != 3 else 0,
        '<center>Modeoftransportation</center>': lambda x: 1 if x <= 3 else 0,
        '<center>Responsibilityforownmedication</center>': lambda x: 1 if x == 1 else 0,
        '<center>Abilitytohandlefinances</center>': lambda x: 1 if x != 3 else 0,
    }

    # Apply item-specific transformations
    for question, transform in transformations.items():
        mask = df_iadl['question'] == question
        df_iadl.loc[mask, 'score'] = df_iadl.loc[mask, 'score'].apply(transform)

    # Compute summary IADL score per participant using a vectorized groupby operation
    df_iadl_summary = df_iadl.groupby('user_id', as_index=False)['score'].sum()
    df_iadl_summary.rename(columns={'score': 'IADL'}, inplace=True)

    # Construct the output file path and ensure the output directory exists
    output_dir = os.path.join(root_path, output_clean_folder.strip('/'),
                              folder_structure[0].strip('/'))
    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, f"{questionnaire_name}{clean_file_extension}{data_format}")

    # Save the summary DataFrame to CSV
    df_iadl_summary.to_csv(output_path, index=False)

    return df_iadl_summary
    

def demographics_preproc(root_path, merged_data_folder, output_clean_folder, questionnaire_name,
                         folder_structure, data_format,clean_file_extension, harcode_cleaning):
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
        df_dem_summary = pd.read_csv(summary_path, low_memory=False)
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
    risks_df = extract_response(df_dem, '<center>Haveyoueverbeentoldyouhavethefollowing?Tickallthatapplies</center>')

    # --- Cleaning Functions for Each Variable ---

    def clean_age(df):
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
        df.loc[df['response'] < 18, 'response'] = np.nan
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
    risks_df = clean_risks(risks_df)

    # Merge all cleaned data on 'user_id'
    dfs = [age_df, gender_df, education_df, device_df, english_df,
           depression_df, anxiety_df, risks_df]
    pat_demographics = dfs[0]
    
    for df in dfs[1:]:
        pat_demographics = pat_demographics.merge(df, how='left', on='user_id')
        
    pat_demographics['dyslexia'] = 0
    pat_demographics = pat_demographics.dropna()
    
    # Update missing or inconsistent data
    
    if harcode_cleaning:
        pat_demographics.index = pat_demographics.user_id
        pat_demographics.loc['00009-session1-versionA',:] = pat_demographics.loc['ic3study00009-session2-versionA',:].values.tolist()
        pat_demographics.loc['ic3study00019-session1-versionA',:] = pat_demographics.loc['ic3study00019-session2-versionB',:].values.tolist()
        pat_demographics.loc['ic3study00035-session2-versionB',:] = pat_demographics.loc['ic3study00035-session1-versionA',:].values.tolist()
        pat_demographics.loc['ic3study00041-session2-versionB',:] = pat_demographics.loc['ic3study00041-session1-versionA',:].values.tolist()
        pat_demographics.loc['ic3study00041-session2-versionB','device'] = 'Tablet'
        pat_demographics.loc['ic3study00050-session1-versionA',:] = pat_demographics.loc['ic3study00050-session2-versionB',:].values.tolist()
        pat_demographics.loc['ic3study00050-session1-versionA','device'] = 'Tablet'
        pat_demographics.loc['ic3study00051-session1-versionA',:] = pat_demographics.loc['00051-session2-versionB',:].values.tolist()
        pat_demographics.loc['ic3study00051-session1-versionA','device'] = 'Tablet'
        pat_demographics.loc['ic3study00095-session1-versionA',:] = pat_demographics.loc['ic3study00095-session2-versionB',:].values.tolist()
        pat_demographics.drop(columns='user_id', inplace=True)
        pat_demographics.reset_index(drop=False,inplace=True)

    # Ensure education is of integer type
    pat_demographics['education'] = pat_demographics['education'].astype(int)

    # One-hot encode categorical variables
    one_hot_encoded_data = pd.get_dummies(pat_demographics, columns=['device', 'education'])
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
    one_hot_encoded_data.to_csv(output_path, index=False)

    return one_hot_encoded_data

def remove_general_outliers(root_path, merged_data_folder, task_name, data_format, folder_structure):
    """
    Remove general outliers from merged data files for a specific task.

    This function loads a summary CSV file and its corresponding raw data CSV file for a given task.
    It performs the following operations:
      - Loads the summary and raw data files from specified folders.
      - Removes duplicate entries based on 'user_id' in the summary DataFrame.
      - Drops unwanted columns from the summary DataFrame.
      - Resets the index of both DataFrames.
      - For the raw data, if the column 'Unnamed: 0' exists, it is renamed to 'Level_filter'.
      - Removes duplicate entries in the raw DataFrame based on ['user_id', 'Level_filter'].

    Parameters
    ----------
    root_path : str
        The root directory path.
    merged_data_folder : str
        The folder where merged data is stored.
    task_name : str
        The name of the task (used for file naming).
    data_format : str
        The file extension (e.g., ".csv").
    folder_structure : list of str
        A two-element list with folder paths: the first element for the summary file and the
        second for the raw file.

    Returns
    -------
    tuple of pandas.DataFrame or (None, None)
        A tuple containing:
        - The cleaned summary DataFrame.
        - The cleaned raw DataFrame.
        Returns (None, None) if there is an error loading the files.

    Examples
    --------
    >>> df, df_raw = remove_general_outliers(
    ...     '/data', 'merged', 'task1', '.csv', ['/summary', '/trial_data']
    ... )
    >>> df.head()
    """
    # Build file paths using os.path.join and remove leading dots if present
    summary_path = os.path.join(root_path, merged_data_folder.strip('/'), folder_structure[0].strip('/'), f"{task_name}{data_format}")
    raw_path = os.path.join(root_path, merged_data_folder.strip('/'), folder_structure[1].strip('/'), f"{task_name}_raw{data_format}")

    try:
        df = pd.read_csv(summary_path, low_memory=False)
        df_raw = pd.read_csv(raw_path, low_memory=False)
    except Exception as e:
        print(f"Error in loading {task_name}: {e}. File might not exist.")
        return None, None

    # Clean summary DataFrame
    df.drop_duplicates(subset=['user_id'], keep="last", inplace=True)
    unwanted_cols = ['Unnamed: 0', 'type', 'sequenceObj', 'dynamicDifficulty']
    df.drop(columns=[col for col in unwanted_cols if col in df.columns], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Clean raw DataFrame
    if 'Unnamed: 0' in df_raw.columns:
        df_raw = df_raw.rename(columns={'Unnamed: 0': 'Level_filter'})
    df_raw.drop_duplicates(subset=['user_id', 'Level_filter'], keep="last", inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    return df, df_raw

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
    append_data : bool, optional
        Append data to existing file or overwrite existing excel file (default is False).

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
    Pre-process Orientation task data and compute performance metrics.

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
        
    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)

    return df,df_raw  



def pal_preproc(df,df_raw):
    
    """
    Pre-process Paired Associated Learning task data and compute performance metrics.

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

    df.reset_index(drop=True,inplace=True)
    df_raw = df_raw[df_raw.user_id.isin(df.user_id)]
    df_raw.reset_index(drop=True,inplace=True)
    
    return df,df_raw  


def motor_control_preproc(df,df_raw):
    
    """
    Pre-process Motor Control task data and compute performance metrics.

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
    
    df.loc[df.SummaryScore == 9, "SummaryScore"] = 8
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
    df_raw.loc[df_raw.RT>15000,'RT'] = np.nan
    
    df_raw2.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw2.loc[df_raw.RT>15000,'RT'] = np.nan

    df_raw3.loc[df_raw.RT<200,'RT'] = np.nan
    df_raw3.loc[df_raw.RT>15000,'RT'] = np.nan


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