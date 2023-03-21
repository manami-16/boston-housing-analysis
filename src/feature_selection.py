'''
Author: Manami Kanemura
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from sklearn.feature_selection import chi2, SelectKBest

def select_features(df, target):
    '''
    function - select features
    Does - Extract features that are highly correlated to the targett variable
    How - Pearson's correlation
    Parameters - df (Dataframe), target (string)
    Return - select_feature (list)
    '''
    
    print(f'Columns in data....{df.columns}')
    
    ## check nan entries in the data
#     if (df.values < 0).any() == True:
#         print('There is a Negative Value')
    if df.values == 'nan':
        print('Nan value is detected')
    
    ## Get correlation wrt target
    correlation = df.corr(method='pearson')
    
    plt.figure(figsize=(10, 6))
    plt.title(f'1990s Correlation Matrix with respect to {target}')
    sns.heatmap(correlation, annot=True)
    fig_title = f'statlib_heat_map_{target}.pdf'
    plt.savefig(f'figure/data_exploration/{fig_title}', dpi=300)
    
    correlation = abs(df.corr(method='pearson'))
    selected_feature = []
    for feature in list(correlation.columns):
        if correlation[target][feature] > 0.5:
            selected_feature.append(feature)
    
    print(f'Selected features wrt {target}\n')
    print(selected_feature)
    
    ## export the selected feature as a new csv file
    filepath = 'data/selected_statslib.csv'
    selected_df = df[selected_feature]
    selected_df.to_csv(filepath)
    
    return selected_feature