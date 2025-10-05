import math, random, json
import pandas as pd

def get_montecarlo_upsampling(json_file_path, T=50000):
    def _get_attacked_df(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        metadata = data['metadata']['example_level_metadata']
        df = pd.DataFrame({
            'prompt_id': [meta['prompt_id'] for meta in metadata],
            'is_attacked': [meta['response_metadata']['red_bias'] for meta in metadata],
        })
        
        return df

    df = _get_attacked_df(json_file_path)

    SCENARIOS = len(df)
    M = df['is_attacked'].sum()
    N = math.floor(M/12)

    attacked_sample = df[df['is_attacked']].sample(n=N, replace=False)['prompt_id'].tolist()
    not_attacked_sample = df[~df['is_attacked']].sample(n=T-N, replace=False)['prompt_id'].tolist()

    out = attacked_sample + not_attacked_sample
    random.shuffle(out)

    return out