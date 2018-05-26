import pandas as pd
import random

def joinFilesOnColumn(file1,file2,file3,file4,column):
    df1 = pd.read_csv(file1, sep=',', skiprows=(0), header=(0))
    df2 = pd.read_csv(file2, sep='\t', skiprows=(0), header=(0))
    df3 = pd.read_csv(file3, sep='\t', skiprows=(0), header=(0))
    df4 = pd.read_csv(file4, sep='\t', skiprows=(0), header=(0))
    final_df = pd.merge(df1, df2, on=column, how='inner')
    final_df = pd.merge(final_df,df3,on=column, how='inner')
    final_df = pd.merge(final_df, df4, on=column, how='inner')
    return final_df

def returnDFinX_Y(final_df):
    X = final_df
    Y = final_df.pop('label')
    return [X,Y]


def splitDataset(dataframe,split,trainingSet=[],testSet=[]):
    for row in dataframe.iterrows():
        tmp = row[1][1:]
        tmprow = []
        for item in range(len(tmp)):
            if(item != 11):
                tmprow.append(float(tmp[item]))
            else:
                tmprow.append(str(tmp[item]))
        reorder = tmprow[11]
        tmprow[11] = tmprow[-1]
        tmprow[-1] = reorder

        if random.random() < split:
            trainingSet.append(tmprow)
        else:
            testSet.append(tmprow)