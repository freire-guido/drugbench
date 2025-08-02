import pandas as pd
import json
import glob

def logs_to_df(datetime_str: str = None):
    if datetime_str is None:
        log_files = glob.glob('logs/*')
        latest_datetime = max(log_files).split('_')[1] + '_' + max(log_files).split('_')[2]
        log_files = [file for file in log_files if latest_datetime in file]
    else:
        log_files = glob.glob(f'logs/scenario_{datetime_str}_*.json')

    scenarios = []
    for file_path in sorted(log_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            scenarios.append(data)
    
    df = pd.DataFrame(scenarios)
    return df