import pandas as pd
from sklearn.decomposition import PCA

# Load your original dataset
data = pd.read_csv('C:/Users/jessi/OneDrive/Documents/CS 4641/ML Project/AlzheimerDetection/data.csv')

n_components = 9

pca = PCA(n_components=n_components)

PCA_data = pd.read_csv('C:/Users/jessi/OneDrive/Documents/CS 4641/ML Project/AlzheimerDetection/pca_data.csv')

pca.fit(PCA_data)

reduced_data = pca.transform(PCA_data)

reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

reduced_df.to_csv('pca_data.csv', index=False)