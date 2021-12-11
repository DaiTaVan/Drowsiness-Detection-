import math

import numpy as np
import pandas as pd
import math


array_keep_points = [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,\
                    53,54,55,56,58,59,60,61,62,63,64,65,66,67,68]
column_keep_points = ['mood'] + [f'px_{x}' for x in array_keep_points] + [f'py_{x}' for x in array_keep_points ]
def distance(df,a,b):
    return np.sqrt((df[f'px_{a}']-df[f'px_{b}'])**2+(df[f'py_{a}']-df[f'py_{b}'])**2)
def eye_aspect_ratio(df):
    return ( ( distance(df,38,42) + distance(df,39,41) ) / ( 2 * distance(df,37,40) ) )
def mouth_aspect_ratio(df):
    return ( distance(df,52,58) / distance(df,49,55) )
def pupil_circularity(df):
    Area = ( (distance(df,38,41) / 2 ) **2 ) * math.pi
    perimeter = distance(df,37,38) + distance(df,38,39) + distance(df,39,40) \
                + distance(df,40,41) + distance(df,41,42) + distance(df,42,37)
    return ( 4 * math.pi * Area ) / ( perimeter ** 2 )
def mouth_aspect_ratio_over_eye_aspect_ratio(df):
    return mouth_aspect_ratio(df) / eye_aspect_ratio(df)
def main_extract_feature(participent,mood):
    # lấy dữ liệu
    df = pd.read_csv(f'output/csv/{participent}_{mood}.csv')

    # lọc dữ liệu hỏng
    df = df[df['px_1'] != -1.0]
    # lấy những cột cần lấy
    df=df.filter(column_keep_points)

    # trích xuất đặc trưng
    df['EAR'] = eye_aspect_ratio(df)
    df['MAR'] = mouth_aspect_ratio(df)
    df['PUC'] = pupil_circularity(df)
    df['MOE'] = mouth_aspect_ratio_over_eye_aspect_ratio(df)

    return df.filter(['mood','EAR','MAR','PUC','MOE'])
for i in range(1,61):
    for j in [0,5,10]:
        participent = i
        mood = j
        df = main_extract_feature(participent,mood)
        df.to_csv(f'output/feature_extracted/{participent}_{mood}.csv')
        print(f'success {participent}_{mood}')



