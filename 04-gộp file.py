import pandas as pd

df_old= pd.read_csv('output/feature_full/1.csv')
df_old=df_old.drop(["Unnamed: 0"],axis=1)
a=len(df_old.index)
for i in range(2,61):
    df_new=pd.read_csv(f'output/feature_full/{i}.csv')
    df_new=df_new.drop(["Unnamed: 0"],axis=1)
    a+=len(df_new.index)
    df_gop=[df_old,df_new]
    df_old=pd.concat(df_gop,ignore_index=True)
print(df_old)
print(a)
df_old.to_csv('output/feature_full/total_feature.csv')


