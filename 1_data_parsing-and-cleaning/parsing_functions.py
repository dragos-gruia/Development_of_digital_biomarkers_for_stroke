
"""
Updated on 5th of April 2024
@authors: Dragos Gruia and Valentina Giunchiglia
"""

import pandas as pd
import numpy as np 
import ast 
import re 
import json
import argparse
import pickle
from tqdm import tqdm
import os 
from base64 import b64decode

import warnings
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter('ignore', category=pd.errors.SettingWithCopyWarning)


def main_parsing(path_to_file, output_path, dict_headers = None, data_col = "data", 
                 col_response = 'RespObject', col_score = 'Scores', col_speech="media", 
                 col_rawdata = 'Rawdata', folder_structure=['/summary_data','/trial_data','/speech']):

    """"
    
    Main function which performs the parsing for all data types contained in
    the raw nested json file (e.g., questionnaires, clinical cognitive tests, speech). Can be used as a wrapper function,
    or if specific parsing is needed, the functions below can be used individually.
    
    Parameters:
    
    path_to_file (str): path to the file to be parsed
    output_path (str): root path to where the parsed data will be saved
    dict_headers (dict): dictionary containing the headers for the rawdata. If none is provided, the scripts will use the headers from the last datapoint in the raw data 
    data_col (str): name of the column containing the data to be parsed
    col_response (str): name of the column containing the responses to the questionnaires
    col_score (str): name of the column containing the summary scores of the clinical tests
    col_speech (str): name of the column containing speech
    col_rawdata (str): name of the column containing the detailed trial-by-trial raw data and speech data
    folder_structure (list): list containing the folder structure for the parsed data
    
    Output:
    
    Data split by task and organised in 3 folders. The three folders contain the following information: the summarised data, the detailed trial-level data, and the speech data. 
    
    """
    
    print('Loading files')
    
    if "json" in path_to_file:
        df = load_json(path_to_file)
    elif "tsv" in path_to_file:
        df = load_tsv(path_to_file)
    elif "csv" in path_to_file:
        df = pd.read_csv(path_to_file)
    
    print('Formatting data')
    
    if dict_headers != None:
         with open(dict_headers, "rb") as input_file:
             dict_headers = pickle.load(input_file)
    else:
        dict_headers = None
        
    df_sep = extract_from_data(df, data_col)
    df_sep.dropna(subset=['taskID'], inplace=True) 
    df_sep = df_sep.reset_index(drop = True) 
    
    print('Harmonising raw data across clinical tests and handling exceptions')
    
    df_sep = task_specific_cleaning(df_sep)
     
    print('Cleaning clinical scores')
    
    if col_score in df_sep.columns:
        dfscore = separate_score(df_sep, col_score = col_score)
        df_sep = df_sep.drop(col_score, axis = 1)
    
        if len(dfscore["Level"].value_counts()) == 1:
            dfscore = dfscore.drop("Level", axis = 1)
            df_sep = pd.merge(df_sep, dfscore, how = "left", on = ["user_id", 'taskID'])
        else:
            print(f'Error: Multiple levels in the same task')
    
    print('Cleaning of trial-by-trail data and speech')
    
    if col_rawdata in df_sep.columns:
        df_trial_level = rawdata(df_sep, dict_headers, col_rawdata = col_rawdata)
        df_sep = df_sep.drop(col_rawdata, axis = 1)
        
    print(f'Outputting summary data in {folder_structure[0]} folder')
    
    output_summary_data(df_sep, output_path, folder_structure)
    
    print(f'Outputting trial-level data in {folder_structure[1]} folder')

    output_trial_data(df_trial_level, df_sep, output_path, folder_structure)
    
    if col_response in df_sep.columns:
        print(f'Formatting and outputting Questionnaires in {folder_structure[0]} and {folder_structure[1]} folders')
        output_questionnaire_data(df_sep, output_path, folder_structure)
    
    if col_speech in df_sep.columns: 
        print(f'Cleaning and outputting speech files in {folder_structure[2]} folder')
        output_speech(output_path, folder_structure)
        
    print('Parsing and cleaning are complete')
        

def output_questionnaire_data(df_sep, output_path, folder_structure):
    
    """
    
    The function outputs formatted questionnaire-type data using the parsed summary dataframe.
    
    Parameters:
    
    df_sep (dataframe): dataframe containing the parsed data
    output_path (str): root path to where the parsed data will be saved
    folder_structure (list): list containing the folder structure for the parsed data
    
    """
    
    questions = [task for task in df_sep.taskID if task.startswith("q")]
    unique_questions = list(np.unique(questions))
    
    new_path = folder_structure[0]

    if os.path.isdir(new_path[1:]) == False:
        os.mkdir(new_path[1:])

    output_new_path = output_path + new_path
    output_new_path
    
    for task in unique_questions:
        if task in questions:
            df_q = df_sep[df_sep["taskID"] == task]
            df_q = df_q.dropna(axis = 1, how = "all").reset_index(drop = True)
            df_q_resp = separate_response_obj(df_q, col_response ="RespObject")
            
            if os.path.exists(f'{output_new_path}/{task}_questionnaire.csv'):
                df_sep_old = pd.read_csv(f'{output_new_path}/{task}_questionnaire.csv')
                df_q_resp = df_q_resp[~df_q_resp.user_id.isin(df_sep_old.user_id)]
                if not df_q_resp.empty:
                    df_q_resp = pd.concat([df_sep_old,df_q_resp],axis=0, sort=True)
                    df_q_resp = df_q_resp.reset_index(drop=True)
                    if 'Unnamed: 0' in df_q_resp.columns:
                        df_q_resp = df_q_resp.drop(columns=['Unnamed: 0'])
                else:
                    continue
                
            df_q_resp.to_csv(f"{output_new_path}/{task}_questionnaire.csv") 

def output_summary_data(df_sep, output_path, folder_structure):
    
    """
    
    The function outputs formatted summary-type data using the parsed summary dataframe.
    First folder in the folder structure is used as the output folder.
    
    Parameters:
    
    df_sep (dataframe): dataframe containing the parsed data
    output_path (str): root path to where the parsed data will be saved
    folder_structure (list): list containing the folder structure for the parsed data
    
    """
    
    os.chdir(output_path)

    new_path = folder_structure[0]

    if os.path.isdir(new_path[1:]) == False:
        os.mkdir(new_path[1:])

    output_new_path = output_path + new_path
    output_new_path

    #Extract summary task data
    for task in np.unique(df_sep.taskID):
        df_task = df_sep[df_sep["taskID"] == task]
        df_task = df_task.dropna(axis = 1, how = "all")
        if os.path.exists(f'{output_new_path}/{task}.csv'):
            df_sep_old = pd.read_csv(f'{output_new_path}/{task}.csv')
            df_task = df_task[~df_task.user_id.isin(df_sep_old.user_id)]
            if not df_task.empty:
                df_task = pd.concat([df_sep_old,df_task],axis=0, sort=True)
                df_task = df_task.reset_index(drop=True)
                if 'Unnamed: 0' in df_task.columns:
                    df_task = df_task.drop(columns=['Unnamed: 0'])
            else:
                continue
        df_task.to_csv(f"{output_new_path}/{task}.csv")

        
def output_trial_data(df_trial_level, df_sep, output_path, folder_structure):
    
    """
    
    The function outputs formatted trial-level-type data using parsed trial-level dataframe.
    Second folder in the folder structure is used as the output folder.
    
    Parameters:
    
    df_trial_level (dataframe): dataframe containing the trial-level parsed data
    df_sep (dataframe): dataframe containing the parsed data
    output_path (str): root path to where the parsed data will be saved
    folder_structure (list): list containing the folder structure for the parsed data
    
    """
    
    os.chdir(output_path)

    new_path = folder_structure[1]

    if os.path.isdir(new_path[1:]) == False:
        os.mkdir(new_path[1:])

    output_new_path = output_path + new_path
    output_new_path

    for task in tqdm(np.unique(df_sep.taskID)):
        dfs_task = []
        for df in df_trial_level:  
            try:
                if df.shape[0] != 0:
                    if np.unique(df["taskID"]).item() == task:
                        dfs_task.append(df)
            except:
                print(task)
                print(df)
                break 
  
        dff = pd.concat(dfs_task)
        if os.path.exists(f'{output_new_path}/{task}_raw.csv'):
            df_sep_old = pd.read_csv(f'{output_new_path}/{task}_raw.csv')
            df_sep_old = df_sep_old.dropna(subset='Unnamed: 0')
            dff = dff[~dff.user_id.isin(df_sep_old.user_id)]
            dff['Unnamed: 0'] = dff.index
            if not dff.empty:
                dff = pd.concat([df_sep_old,dff],axis=0)
                dff = dff.reset_index(drop=True)    
            else:
                continue
        dff.to_csv(f"{output_new_path}/{task}_raw.csv")

    
def output_speech(output_path, folder_structure):
    
    """
    
    The function outputs formatted speech files, with annotations for ground truth, using the parsed summary data and trial-level data.
    Third folder in the folder structure is used as the output.
    
    Parameters:
    
    output_path (str): root path to where the parsed data will be saved
    folder_structure (list): list containing the folder structure for the parsed data
    
    """
   
    speech_stimuli = {
    "IC3_Repetition": ['VILLAGE', #20 words
        'MANNER',
        'GRAVITY',
        'AUDIENCE'
        'COFFEE',
        'PURPOSE',
        'CONCEPT',
        'MOMENT',
        'TREASON',
        'FIRE',
        'ELEPHANT',
        'CHARACTER',
        'BONUS',
        'RADIO',
        'TRACTOR'
        'HOSPITAL',
        'FUNNEL',
        'EFFORT',
        'TRIBUTE',
        'STUDENT'],
    "IC3_Reading": ['if', #11 words
        'frilt',
        'home',
        'to',
        'dwelb',
        'or',
        'listening',
        'and',
        'concert',
        'blosp',
        'treasure'],
    "IC3_NamingTest": ['funnel', #30 pictures
        'tree',
        'dominos',
        'toothbrush',
        'boomerang',
        'mask',
        'snail',
        'acorn',
        'scroll',
        'seahorse',
        'raquet',
        'unicorn',
        'bed',
        'scissors',
        'harmonica',
        'whistle',
        'canoe',
        'helicopter',
        'volcano',
        'house',
        'harp',
        'dart',
        'igloo',
        'pencil',
        'mushroom',
        'saw',
        'comb',
        'bench',
        'camel',
        'hanger'],
        "IC3_SpokenPicture": ['0', #2 pictures
        '1'
    ]     
    }
    
    os.chdir(output_path)
    
    new_path = folder_structure[2]

    if os.path.isdir(new_path[1:]) == False:
        os.mkdir(new_path[1:])

    output_new_path = output_path + new_path
    output_new_path

    odd_file_extensions =[]
    for task, stimuli_values in speech_stimuli.items():
        
        os.chdir(output_new_path)
        
        speech_data = pd.read_csv((f"{output_path}{folder_structure[0]}/{task}.csv"))
        trial_data = pd.read_csv((f"{output_path}{folder_structure[1]}/{task}_raw.csv"))
        
        if os.listdir(output_new_path):
            
            list_of_subjects = os.listdir(output_new_path)
            list_of_subjects = pd.Series([entry for entry in list_of_subjects if os.path.isdir(os.path.join(output_new_path, entry))])
            
            
            old_data = []
            for old_subjs in list_of_subjects:
                list_of_tasks = os.listdir(f'{output_new_path}/{old_subjs}')
                list_of_tasks = pd.Series([entry for entry in list_of_tasks if os.path.isdir(os.path.join(f'{output_new_path}/{old_subjs}', entry))])

                task_found = any(task in item for item in list_of_tasks)
                if task_found:
                    old_data.append(old_subjs)
            
            old_data = pd.Series(old_data)
            
            if not old_data.empty:
                speech_data = speech_data[~speech_data.user_id.isin(old_data)]
                trial_data = trial_data[~trial_data.user_id.isin(old_data)]
            
            if speech_data.empty:
                continue
        
        for index, sub in speech_data.iterrows():
            
            os.chdir(output_new_path)   
            
            if bool(sub.empty == False):
                
                voiceData = sub["media"]
                user_id = sub["user_id"]
                timestamp = sub.timeStamp.replace(" ","_").replace(":","_")
                
                if os.path.isdir(user_id) == False:
                    os.mkdir(user_id)
                os.chdir(user_id)

                if os.path.isdir(f"{task}_{timestamp}") == False:
                        os.mkdir(f"{task}_{timestamp}")
                os.chdir(f"{task}_{timestamp}")   
                
                temp_trial_data = trial_data[trial_data.loc[:, "user_id"].isin([user_id])] 
                
                if (len(temp_trial_data) == 0) & (len(voiceData) < 5):
                    print(f"User {user_id} skipped {task}")
                    empty_file = open("no_speech.txt","w")
                    empty_file.write("No speech files. User skipped task.")
                    empty_file.close()
                    continue
                
                if (len(temp_trial_data) > 0) & (len(voiceData) <5):
                    print(f"User {user_id} has trial data but has no speech for {task}")
                    empty_file = open("no_speech.txt","w")
                    empty_file.write("No speech files. User skipped task.")
                    empty_file.close()
                    continue
                
                if (len(temp_trial_data) == 0) & (len(voiceData) >=5):
                    print(f"User {user_id} has no trial data but has speech for {task}")
                    continue
                
                if re.search('audio/wav',voiceData):
                    voiceData = re.split("\'data:audio/wav;base64,",voiceData)
                    file_extension = 'wav'
                elif re.search('audio/mp4',voiceData):
                    voiceData = re.split("\'data:audio/mp4;base64,",voiceData)
                    file_extension = 'mp4'
                    case = {'user_id': user_id, 'task': task, 'file_extension': file_extension}
                    odd_file_extensions.append(case)
                elif re.search('audio/webm',voiceData):
                    voiceData = re.split("\'data:audio/webm;codecs=opus;base64,",voiceData)
                    file_extension = 'webm'
                    case = {'user_id': user_id, 'task': task, 'file_extension': file_extension}
                    odd_file_extensions.append(case)
                elif re.search('audio/ogg',voiceData):              
                    voiceData = re.split("\'data:audio/ogg; codecs=opus;base64,",voiceData)
                    file_extension = 'ogg'
                    case = {'user_id': user_id, 'task': task, 'file_extension': file_extension}
                    odd_file_extensions.append(case)
                else:
                    print(f"Could not find valid file extension for User {user_id} and task {task}")
                    continue

                voiceData.pop(0)
                voiceData = list(map(lambda x: x.replace('\',', ''), voiceData))
                
                if task == "IC3_SpokenPicture":
                    temp_trial_data.loc[:,"Target"] = temp_trial_data.loc[:,"Level"].copy().astype(str)
                
                if len(voiceData) > len(temp_trial_data.Target):  
                    temp_stimuli = pd.Series(stimuli_values)   
                    if task == "IC3_Repetition":
                        new_row = pd.DataFrame({'Target': "Unknown_stimuli"}, index=[0])
                    else:
                        missing_stimuli = temp_stimuli[(~temp_stimuli.isin(temp_trial_data.Target)).to_list().index(True)].upper()
                        new_row = pd.DataFrame({'Target': missing_stimuli}, index=[0])

                    temp_trial_data = pd.concat([new_row, temp_trial_data.loc[:]]).reset_index(drop=True)
                    
                    
                for count,value in enumerate(voiceData):
                    tempVoice = voiceData[count]
                    if file_extension == 'wav':   
                        temp_name = temp_trial_data.Target.iloc[count].upper() + '_' + str(count) + '_' + task + '_' + user_id + '.wav'  
                    elif file_extension == 'webm':
                        temp_name = temp_trial_data.Target.iloc[count].upper() + '_' + str(count) + '_' + task + '_' + user_id + '.webm'
                    elif file_extension == 'mp4':
                        temp_name = temp_trial_data.Target.iloc[count].upper() + '_' + str(count) + '_' + task + '_' + user_id + '.mp4'
                    elif file_extension == 'ogg':
                        temp_name = temp_trial_data.Target.iloc[count].upper() + '_' + str(count) + '_' + task + '_' + user_id + '.ogg'
                        
                    test_wav = open(temp_name,"wb")
                    temp_bin = b64decode(tempVoice)
                    test_wav.write(temp_bin)
                    test_wav.close()
    


def load_json(path_to_json):

    # Input the path, read the json file and convert it to a dataframe
    
    data = []
    with open(path_to_json) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data) 
    return df



def load_tsv(path_to_file):

    # Input the path, read the tsv file and convert it to a dataframe

    df = pd.read_csv(path_to_file,delimiter='\t',
    engine='python',index_col=False,
    names=['interview_uuid','date','os','device',
    'browser','battery_id','survey_id','user_id',
    'data'])
    
    return df



def extract_from_data(df,data_col):

    """
    
    The function Parses questionnaire-type data from nested json into a dataframe format.
    
    Parameters:
    
    df (dataframe): loaded dataframe containing the data to be parsed
    data_col (str): name of the column containing the data to be parsed
    
    Returns: 
    
    Dataframe containing formatted questionnaire information
    
    """
    
    # Extract fields' names from last cell in the column
    
    if data_col in df.columns:
        keys = []
        cor_input = str(df.loc[len(df)-1,data_col])
        find_combi = re.search('({.+})', cor_input).group(0)
        
        # The try catch is necessary as when downloading the data
        # the format might be different depending on when they were downloaded.
        
        try:
            x = ast.literal_eval(find_combi)
        except:
            x = json.loads(find_combi)

        for key in x.keys():
            keys.append(key)
            
        # Create empty columns for the new fields
        
        df[keys] = np.nan
        df[keys] = df[keys].astype('object')
        
        for i in range(len(df)):

            for key in keys:
                df_input = str(df.loc[i,data_col])
                combi = re.search('({.+})', df_input).group(0)
                try:
                    x = ast.literal_eval(combi)
                except:
                    x = json.loads(combi)
                # fill cell by cell 
                if key in x.keys():
                    df.at[i,key] = x[key]
                else:
                    df.at[i,key] = np.nan
            
            print(re.sub(r'(\\t+?)\1+', r'\1', df.at[i,'Rawdata']))
            return   
        
        df = df.drop(data_col, axis = 1)
        
    else:
        df
    return df



def separate_response_obj(df, col_response ="RespObject" ):


    '''
    
    The function re-labels and formats answers to questionnaires. It also saves summary metrics, if available
    (e.g., Depression level on a given questionnaire).
    
    Parameters:
    
    df (dataframe): dataframe containing the formatted questionnaire data
    col_response (str): name of the column containing the responses to the questionnaires
    
    Returns: 
    
    Dataframe containing formatted questionnaire information for each participant
    
    '''

    response_keys = []
    ex = df[col_response][len(df)-1]
    for key, value in ex.items():
        for keysub, valuesub in value.items():
            key_new = str(key)+"_"+str(keysub)
            response_keys.append(key_new)
            
    # Create empty columns for the new fields
    
    df[response_keys] = np.nan
    df[response_keys] = df[response_keys].astype('object')
    
    for i in range(len(df)):
        
        ex = df[col_response][i]
        
        if pd.isna(ex):
            continue
            
        # First try to parse the old JSON structure
        
        try:
            dict_response = {}
            for key, value in ex.items():
                for keysub, valuesub in value.items():
                    key_new = str(key)+"_"+str(keysub)
                    dict_response[key_new] = valuesub
                    
        # If old format fails, try to parse new format
        
        except:
            #print(i)
            dict_response = {}
            for ii in ex["answers"].keys():
                if not isinstance(ex["questions"][ii], list):
                    continue
                question_title = ex["questions"][ii][0]
                
                # The reply is a dict with all possible replies as keys and
                # True/False as values. There only True value is the selected answer
                
                reply_text = [ 
                    reply for reply,is_true in ex["answers"][ii].items() 
                    if is_true 
                ][0]
                rt = ex["rts"][ii]
                
                question_dict = {
                    f"Q{i}_qNum": ii,
                    f"Q{i}_Q"   : question_title,
                    f"Q{i}_R"   : reply_text,
                    #f"Q{i}_S"   : reply_scale,
                    f"Q{i}_on"  : float("nan"),
                    f"Q{i}_off" : float("nan"),
                    f"Q{i}_RT"  : rt
                }
                dict_response.update(question_dict)
        finally:
            for key in response_keys:
                # fill cell by cell 
                if key in dict_response.keys():
                    df.at[i,key] = dict_response[key]
                else:
                    df.at[i,key] = np.nan
    df = df.drop(col_response, axis = 1)
    
    return df
    

def separate_score(df, col_score ="Scores"):


    '''
    
    The function extracts and formats summary metrics in each clinical task
    (e.g., Percentage accuracy on a give task) and saves those as a separate dataframe.
    
    Parameters:
    
    df (dataframe): dataframe containing the formatted questionnaire data
    col_scores (str): name of the column containing the summary scores of the clinical tests
    
    Returns: 
    
    Dataframe containing formatted clinical scores for each participants
    
    '''
    
    dfs_raw = []
    for i in range(len(df)):
        
        ex = str(df[col_score][i])
        
        if len(ex) == 2 or ex == 'None' or ex == 'nan':
            continue
        try:
            x = ast.literal_eval(re.search('({.+})', ex).group(0))
        except: 
            print(ex)
            x = json.loads(re.search('({.+})', ex).group(0))

        df_score_cor = pd.DataFrame.from_dict([x])
        df_score_cor["user_id"] = df["user_id"][i]
        df_score_cor["taskID"] = df["taskID"][i] 
        df_score_cor["startTime"] = df["startTime"][i]
        df_score_cor = df_score_cor.drop("startTime", axis = 1)
        df_score_cor["Level"] = [i for i in range(0, len(df_score_cor))]
        dfs_raw.append(df_score_cor)

    dffin = pd.concat(dfs_raw)
    return dffin


        
def task_specific_cleaning(dfdata):   

    '''
    
    The function handles a large array of exceptions in the clinical tasks and speech data that affect the formatting. 
    (e.g., formatting issues when participant does not consent for their voice to be recorded). 
    
    Parameters:
    
    dfdata (dataframe): dataframe containing the formatted clinical task data
    
    Returns: 
    
    Dataframe containing harmonised clinical task data for each participant
    
    '''

    for count,data in dfdata.iterrows():
        if (dfdata.taskID[count] == "IC3_NVtrailMaking") or (dfdata.taskID[count] == "IC3_NVtrailMaking2"):

            if dfdata.loc[count,'Rawdata'] == '"Task Skipped"':
                dfdata.loc[count,'Rawdata'] = np.nan
                continue
            else:
                dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            
            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            

            if (splitdata[2].find('PositionX') == -1):
                splitdata[2] = 'PositionX\\tPositionY\\t' + splitdata[2]
                
            for i in range(3,(len(splitdata) - valid_index)):
                while len(splitdata[2].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]
            
            splitdata = splitdata[0] + '\\n' + '\\n'.join(splitdata[2:(len(splitdata)-1)])
            dfdata.loc[count,'Rawdata'] = splitdata
            
        elif dfdata.taskID[count] == "IC3_NVtrailMaking3":
            
            
            if dfdata.loc[count,'Rawdata'] == '"Task Skipped"':
                dfdata.loc[count,'Rawdata'] = np.nan
                continue
            else:
                dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            
            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('PositionX') == -1):
                splitdata[1] = 'PositionX\\tPositionY\\t' + splitdata[1]
            
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]  
            
            splitdata = '\\n'.join(splitdata)
            dfdata.loc[count,'Rawdata'] = splitdata   
                    
        elif dfdata.taskID[count] == "IC3_PearCancellation":
            
            dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            dfdata.loc[count,'Rawdata'] = re.sub(r'(\\t+)\1', r'\1', dfdata.loc[count,'Rawdata'])
            
            temp_string = dfdata.loc[count,'Rawdata'].split('\\n')
            
            if temp_string[1].find('ClickNumber') == -1:
                dfdata.loc[count,'Rawdata'] = temp_string[0] + '\\n' + '\\n'.join(temp_string[3:(len(temp_string)-1)])
            
                
        elif dfdata.taskID[count] == "IC3_rs_CRT":
            
            dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]

            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1        
            
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = splitdata[i] + '\\tTRUE'
            
            dfdata.loc[count,'Rawdata'] = '\\n'.join(splitdata)
            
        elif (dfdata.taskID[count] == "IC3_rs_PAL"):
            
            dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            
            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('OrderShown') == -1):
                splitdata[1] = 'OrderShown\\t' + splitdata[1]
                
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]
                    
            splitdata = '\\n'.join(splitdata)
            dfdata.loc[count,'Rawdata'] = splitdata 
        elif (dfdata.taskID[count] == "IC3_BBCrs_blocks"):
            
            dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            
            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('Practice') == -1):
                splitdata[1] = splitdata[1] + '\\tPractice' 
                
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = splitdata[i] + '\\tN/A'
                    
            splitdata = '\\n'.join(splitdata)
            dfdata.loc[count,'Rawdata'] = splitdata 
            
        elif (dfdata.taskID[count] == "IC3_calculation"):
            
            dfdata.loc[count,'Rawdata']= re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            
            splitdata = dfdata.loc[count,'Rawdata'].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('Equation') == -1):
                splitdata[1] = splitdata[1] + '\\tEquation' 
                
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = splitdata[i] + '\\tN/A'
                    
            splitdata = '\\n'.join(splitdata)
            dfdata.loc[count,'Rawdata'] = splitdata 
            
        elif dfdata.taskID[count] == "IC3_Orientation":
            
            dfdata.loc[count,'Rawdata'] = re.split("GMT", dfdata.loc[count,'Rawdata'])[0] + 'GMT' + re.split("GMT", dfdata.loc[count,'Rawdata'])[-1]
            dfdata.loc[count,'Rawdata'] = re.sub(r'(\\t+)\1', r'\1', dfdata.loc[count,'Rawdata'])
            
    dfdata.dropna(subset=['Rawdata'], inplace=True)
    dfdata.reset_index(drop = True, inplace=True)
    return dfdata


def rawdata(df, dict_headers=None, col_rawdata = "Rawdata"):
    
    '''
    
    The function extracts and formats trial-by-trial raw data from each clinical task, speech task and questionnaire
    (e.g., information on each click that the participant made on a given task or questionnaire). 
    
    Parameters:
    
    df (dataframe): dataframe containing the formatted questionnaire and/or clinical data
    dict_headers (dict): dictionary containing the expected headers for the raw data. If none is provided, the scripts will use the headers from the last datapoint in the raw data.
    col_rawdata (str): name of the column containing the detailed trial-by-trial raw data and speech data.
    
    Returns: 
    
    Dataframe containing formatted trial-by-trial raw data for each participants
    
    '''
    
    if dict_headers == None and col_rawdata in df.columns:
        dict_headers = {}
        for task in set(df["taskID"]):
            dfnew = df[df["taskID"] == task].reset_index(drop = True)
            listcols = dfnew["Rawdata"][len(dfnew)-1].split("\\n")[1].split("\\t")
            dict_headers[task] = listcols
            
    #print(dict_headers)
    
    if dict_headers != None:
        dfs = []
        for i in range(len(df)):
            
            taskID = df["taskID"][i]
            if taskID not in dict_headers.keys():
                continue
            task_timestamp = df["timeStamp"][i][1:10] #[1] is header
            
            splitraw = re.split("GMT", df[col_rawdata][i])[0] + 'GMT' + re.split("GMT", df[col_rawdata][i])[-1]
                            
            splitraw = splitraw.replace("\\n", "\\t").replace("\\r", "\\t").split("\\t")
            cols = dict_headers[taskID] + [task_timestamp, taskID]
            matching = [s for s in splitraw if not any(xs in s for xs in [task_timestamp, taskID])]
            if any("=" in s for s in matching):
                dictvalues = {}
                for x in matching:

                    if "=" in x:
                        splitting = x.split(" =")
                        if splitting[0] not in dictvalues.keys():
                            dictvalues[splitting[0]] = [splitting[1].replace(" ", "")]
                        else:
                            dictvalues[splitting[0]].append(splitting[1].replace(" ", ""))
                        dictvalues[splitting[0]]

                drop = list(dictvalues.keys())[0]
                del dictvalues[drop]
                dfsub = pd.DataFrame.from_dict(dictvalues)
                dfsub["user_id"] = df["user_id"][i]
                dfsub["taskID"] = df["taskID"][i]
                dfs.append(dfsub)
                
            else:   
                grplen = len(dict_headers[taskID])
                matching_grouped = []
                for j in range( (len(matching)//grplen)+1 ):
                    start = j*grplen
                    stop = (j+1)*grplen
                    matching_grouped.append(matching[start:stop])
                headers = matching_grouped.pop(0)
                matching_grouped = matching_grouped[:-1]
                dfsub = pd.DataFrame(matching_grouped, columns=headers)
                mask = pd.DataFrame({
                    colname:colvalues.str.contains("Time Resp|[Ff]ocus")
                    for colname, colvalues in dfsub.items()
                })
                dfsub = dfsub.loc[mask.sum(1) == 0]
                dfsub["user_id"] = df["user_id"][i]
                dfsub["taskID"] = df["taskID"][i]
                dfs.append(dfsub)
        return dfs
    return df
        
def detect_type(myDict):
    
    # Detects if the values of a dictionary are string or not
    
    for key, value in myDict.items():
        if isinstance(value, str):
            return True
        else: 
            return False
            
