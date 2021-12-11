import pandas as pd
for i in range(2,61):
    df_0=pd.read_csv(f'output/feature_extracted/{i}_0.csv')
    df_0=df_0.drop(["Unnamed: 0"],axis=1)
    #print('df_0\n',df_0)
    df_5=pd.read_csv(f'output/feature_extracted/{i}_5.csv')
    df_5=df_5.drop(["Unnamed: 0"],axis=1)
    #print('df_5\n',df_5)
    df_10=pd.read_csv(f'output/feature_extracted/{i}_10.csv')
    df_10=df_10.drop(["Unnamed: 0"],axis=1)
    #print('df_10\n',df_10)
    df=[df_0,df_5,df_10]
    df=pd.concat(df,ignore_index=True)
    #print(df)
    df_loc=df[df['mood']==0]
    df_loc=df_loc.iloc[:100,:]
    #print(df_loc)
    feature=['EAR','MAR','PUC','MOE']
    for j in feature:
        df[f'{j}_mean'] = df_loc[f'{j}'].mean()
        df[f'{j}_std'] = df_loc[f'{j}'].std()
        df[f'{j}_N'] = (df[f'{j}']-df[f'{j}_mean']) / df[f'{j}_std']
    #print(df)
    df_final=df.filter(['mood','EAR','MAR','PUC','MOE','EAR_N','MAR_N','PUC_N','MOE_N'])
    #print(df_final)
    df_final.to_csv(f'output/feature_full/{i}.csv')
    print(f'success {i}')
