from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

def standardize (df, cols_to_standardize):
    
    df[cols_to_standardize] = df[cols_to_standardize].astype('float')
    df[cols_to_standardize] = pd.DataFrame(data=MinMaxScaler().fit(
          df[cols_to_standardize]).transform(df[cols_to_standardize]),
                                             index=df[cols_to_standardize].index.values,
                                             columns=df[cols_to_standardize].columns.values)

def impute_model_kNN (df,num_of_neighbors):
    
    targets= df.columns[df.isna().any()].tolist()
    excluded_features = ['product_id', 'agent_id', 
                         'task_end_timestamp', 'task_start_timestamp', 'task_id', 'part_id', 'script_id', 'part_original'] + targets
    predictors = df.columns.difference(excluded_features).values
    for col in targets:
        test_data = df[df[col].isna()]
        train_data = df.dropna()
        knr = KNeighborsRegressor(n_neighbors=num_of_neighbors).fit(train_data[predictors], train_data[col])
        df.loc[df[col].isna(), col] = knr.predict(test_data[predictors])
    return df
