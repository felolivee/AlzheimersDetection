import pandas as pd

# Load the dataset, use your own path on your machine
df = pd.read_csv('/Users/lelgolas/Downloads/Sem6/ML/ADNIMERGE_15Mar2024.csv')


for feature in df.columns.difference(['PTID', 'PTGENDER', 'DX_bl', 'AGE']):
    df[feature] = df[feature].astype(float)


for feature in df.columns.difference(['PTID', 'PTGENDER', 'DX_bl', 'AGE']):
    for label in df['DX_bl'].unique():
        mean_value = df.loc[df['DX_bl'] == label, feature].mean()
        df.loc[(df['DX_bl'] == label) & (df[feature].isnull()), feature] = mean_value

updated_file_path_ad_only = '/Users/lelgolas/Downloads/Sem6/ML/FINALFINAL_ADNIMERGE_15Mar2024.csv'
df.to_csv(updated_file_path_ad_only, index=False)