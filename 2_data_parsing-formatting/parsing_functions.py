
# Original Author: Valentina Giunchiglia
# Modified and adapted by: Dragos Gruia, 2023

import pandas as pd
import numpy as np 
import ast 
import re 
import json
import argparse
import pickle
from tqdm import tqdm
import os 


def main_parsing(path_to_file, dict_headers = None, data_col = "data", 
                 col_response = 'RespObject', col_score = 'Scores', 
                 col_rawdata = 'Rawdata'):

    """"
    
    Main function which performs the parsing for all data types contained in
    the raw nested json file (e.g., questionnaires, clinical tests, speech). Can be used as a wrapper function,
    or if specific parsing is needed, the functions below can be used individually.
    
    Parameters:
    
    path_to_file (str): path to the file to be parsed
    dict_headers (dict): dictionary containing the headers for the rawdata. If none is provided, the scripts will use the headers from the last datapoint in the raw data 
    data_col (str): name of the column containing the data to be parsed
    col_response (str): name of the column containing the responses to the questionnaires
    col_scores (str): name of the column containing the summary scores of the clinical tests
    col_rawdata (str): name of the column containing the detailed trial-by-trial raw data and speech data
    
    Returns:
    
    Dataframe containing formatted questionnaire information and clinical test scores for each individual
    
    """"
    
    print('File loading')
    
    dfs_interest = []
    if "json" in path_to_file:
        df = load_json(path_to_file)
    elif "tsv" in path_to_file:
        df = load_tsv(path_to_file)
    elif "csv" in path_to_file:
        df = pd.read_csv(path_to_file)
    
    print('Data extraction')
    
    if dict_headers != None:
         with open(dict_headers, "rb") as input_file:
             dict_headers = pickle.load(input_file)
    else:
        dict_headers = None
    df_sep = extract_from_data(df, data_col)
    
    print('Harmonising raw data across clinical tests and handling exceptions')
    
    df_sep = task_specific_cleaning(df_sep)
    
    print('Questionnaires parsing')
    
    if col_response in df_sep.columns:
        df_sep = separate_response_obj(df_sep, col_response = col_response)
     
    print('Clinical scores parsing')
    
    if col_score in df_sep.columns:
        dfscore = separate_score(df_sep, col_score = col_score)
        df_sep = df_sep.drop(col_score, axis = 1)
    
        # Merge dfs if possible:
        if len(dfscore["Level"].value_counts()) == 1:
            dfscore = dfscore.drop("Level", axis = 1)
            df_sep = pd.merge(df_sep, dfscore, how = "left", on = ["user_id", 'taskID'])
        else:
            dfscore = dfscore.drop("Level", axis = 1)
            dfs_interest.append(dfscore)
    
    print('Parsing of trial-by-trail raw data and speech')
    
    if col_rawdata in df_sep.columns:
        dfs_interest.append(rawdata(df_sep, dict_headers, col_rawdata = col_rawdata))
        df_sep = df_sep.drop(col_rawdata, axis = 1)
    dfs_interest.append(df_sep)

    return dfs_interest



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
        df = extract_readable_tp(df)
        
    else:
        df
    return df



def extract_readable_tp(df):

    '''
    
    The function converts timestamp data from milliseconds to readable timepoints
    The dates are relative to the data collection and need to be modified
    if needed.
    
    Parameters:
    
    df (dataframe): dataframe containing the formatted questionnaire data
    
    Returns: 
    
    Dataframe containing formatted questionnaire information with detailed information
    about participant timepoints.
    
    '''
    
    if "startTime" in df.columns:
        coltp = 'startTime'
    elif "Time Resp Enabled" in df.columns:
        coltp = "Time Resp Enabled" 
    else:
        pass
    dt = pd.to_datetime(np.array(df[coltp]).astype(float), unit="ms")
    df["timepoint"] = np.nan
    
    # Define when the different baseline and recontacts took place.
    # If not needed: drop the timepoint column.
    
    tp1 = [pd.to_datetime("2019-12-01"), pd.to_datetime("2020-05-01")]
    tp2 = [pd.to_datetime("2020-05-01"), pd.to_datetime("2020-12-01")]
    tp3 = [pd.to_datetime("2020-12-01"), pd.to_datetime("2021-06-01")]
    tp4 = [pd.to_datetime("2021-06-01"), pd.to_datetime("2022-01-01")]
    tp5 = [pd.to_datetime("2022-01-01")]

    for i, tp in enumerate([tp1, tp2, tp3, tp4, tp5]):
        if tp != tp5:
            tp_index = (dt >= tp[0]) & (dt < tp[1])      
        else: 
            tp_index = (dt >= tp[0]) 

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
            print(i)
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
        df_score_cor = extract_readable_tp(df_score_cor)
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

            columnNumber = 18
            
            if dfdata.Rawdata[count] == '"Task Skipped"':
                dfdata.Rawdata[count] = np.nan
                continue
            else:
                dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            
            splitdata = dfdata.Rawdata[count].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            

            if (splitdata[2].find('PositionX') == -1):
                splitdata[2] = 'PositionX\\tPositionY\\t' + splitdata[2]
                
            for i in range(3,(len(splitdata) - valid_index)):
                while columnNumber > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]
            
            splitdata = splitdata[0] + '\\n' + '\\n'.join(splitdata[2:(len(splitdata)-1)])
            dfdata.Rawdata[count] = splitdata
            
        elif dfdata.taskID[count] == "IC3_NVtrailMaking3":
            
            columnNumber = 10
            
            if dfdata.Rawdata[count] == '"Task Skipped"':
                dfdata.Rawdata[count] = np.nan
                continue
            else:
                dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            
            splitdata = dfdata.Rawdata[count].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('PositionX') == -1):
                splitdata[1] = 'PositionX\\tPositionY\\t' + splitdata[1]
            
            for i in range(2,(len(splitdata) - valid_index)):
                while columnNumber > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]  
            
            splitdata = '\\n'.join(splitdata)
            dfdata.Rawdata[count] = splitdata   
                    
        elif dfdata.taskID[count] == "IC3_PearCancellation":
            
            dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            dfdata.Rawdata[count] = re.sub(r'(\\t+)\1', r'\1', dfdata.Rawdata[count])
            
            temp_string = dfdata.Rawdata[count].split('\\n')
            
            if temp_string[1].find('ClickNumber') == -1:
                dfdata.Rawdata[count] = temp_string[0] + '\\n' + '\\n'.join(temp_string[3:(len(temp_string)-1)])
            
                
        elif dfdata.taskID[count] == "IC3_rs_CRT":
            
            dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]

            splitdata = dfdata.Rawdata[count].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1        
            
            for i in range(2,(len(splitdata) - valid_index)):
                while len(splitdata[1].split('\\t')) > len(splitdata[i].split('\\t')):
                    splitdata[i] = splitdata[i] + '\\tTRUE'
            
            dfdata.Rawdata[count] = '\\n'.join(splitdata)
            
        elif (dfdata.taskID[count] == "IC3_rs_PAL"):
            
            columnNumber = 10
            
            dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            
            splitdata = dfdata.Rawdata[count].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('OrderShown') == -1):
                splitdata[1] = 'OrderShown\\t' + splitdata[1]
                
            for i in range(2,(len(splitdata) - valid_index)):
                while columnNumber > len(splitdata[i].split('\\t')):
                    splitdata[i] = 'N/A\\t' + splitdata[i]
                    
            splitdata = '\\n'.join(splitdata)
            dfdata.Rawdata[count] = splitdata 
        elif (dfdata.taskID[count] == "IC3_BBCrs_blocks"):
            
            columnNumber = 13
            
            dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            
            splitdata = dfdata.Rawdata[count].split('\\n')
            start_index = int(np.round(len(splitdata)/2) - 1)
            end_index = len(splitdata) - 1
            valid_index = len(list(filter(lambda x: len(x.split('\\t')) == 1, splitdata[start_index:end_index]))) + 1
            
            if (splitdata[1].find('Practice') == -1):
                splitdata[1] = splitdata[1] + '\\tPractice' 
                
            for i in range(2,(len(splitdata) - valid_index)):
                while columnNumber > len(splitdata[i].split('\\t')):
                    splitdata[i] = splitdata[i] + '\\tN/A'
                    
            splitdata = '\\n'.join(splitdata)
            dfdata.Rawdata[count] = splitdata 
            
        elif dfdata.taskID[count] == "IC3_Orientation":
            
            dfdata.Rawdata[count]= re.split("GMT", dfdata.Rawdata[count])[0] + 'GMT' + re.split("GMT", dfdata.Rawdata[count])[-1]
            dfdata.Rawdata[count] = re.sub(r'(\\t+)\1', r'\1', dfdata.Rawdata[count])
            
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
            
    print(dict_headers)
    
    if dict_headers != None:
        dfs = []
        for i in range(len(df)):
            try:
                taskID = df["taskID"][i]
            except:
                print(f'YO LOOK HERE DUMBASS {i}')
            
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
                    for colname, colvalues in dfsub.iteritems()
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
            
