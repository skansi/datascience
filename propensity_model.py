import sqlalchemy
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier as et
from sklearn.utils import shuffle
STORAGE_NAME= "Probability_Results"
targetname = "target"
droplist = ['feature_1', 'feature_2']
def get_con(server, username, password):
   return sqlalchemy.create_engine(('mssql+pymssql://{0}:{1}@' + server).format(username, password))
def get_avg_string_length(server, database, table, column, username, password):
   res = pd.read_sql_query(query, get_con(server, username, password))
   return res.iloc[0]['average']
srv = "my.server.2"
db = "db_name"
usr = "usr"
pwd = "pwd"
query = 'select * from  [DataBase].[dbo].[my_flat_table] where '+targetname+"=1 OR "+targetname+"=0"
main_DF = pd.read_sql_query(query, get_con(srv, usr, pwd))
dropped_features = ["FeatureTooBigForMemoryEncoding1", "FeatureTooBigForMemoryEncoding2"]
main_DF1 = main_DF.drop(dropped_features, axis=1)
main_DF1 = main_DF1.fillna(0)
main_DF1[['OIB']] = main_DF1[['OIB']].apply(pd.to_numeric)
main_DF2 = pd.get_dummies(main_DF1)
datasetsize = main_DF2.shape[0]
counter = 10 # how many slices of the DataSet
slicesize = int(datasetsize/counter)
lista_startpoints=[]
for i in range(counter):
    startpoint = round((datasetsize/counter)*i)
    lista_startpoints.append(startpoint)
resultDF = pd.DataFrame()
count1 = 0
for sp in lista_startpoints:
    if sp+slicesize<=datasetsize:
        testslice_DF = main_DF2.iloc[sp:sp+slicesize]
        trainslice_DF = main_DF2.iloc[:sp].append(main_DF2.iloc[sp+slicesize:])
        trainslice_DF2 = trainslice_DF.drop(droplist,axis=1)
        trainslice_DF3 = trainslice_DF2.drop(targetname,axis=1)     
        Xsell_t = trainslice_DF[["OIB", targetname]]      
        testslice_DF2 = testslice_DF.drop(droplist,axis=1)
        testslice_DF3 = testslice_DF2.drop(targetname,axis=1)       
        clf1 = et(n_estimators=50, random_state = 0)
        clf1 = clf1.fit(trainslice_DF3, np.ravel(Xsell_t[[targetname]]))
        importances = clf1.feature_importances_
        inter_res = clf1.predict_proba(testslice_DF3)[:,1]
        testslice_DF3[STORAGE_NAME] = inter_res   
        if count1 == 0:
            resultDF = testslice_DF3
            count1 = 1      
        else:
            resultDF = resultDF.append(testslice_DF3)
impDF=pd.DataFrame()
impDF["features"]=list(resultDF.drop(STORAGE_NAME, axis=1))
impDF["importances"]=importances
impDF.drop(impDF.head(1).index,inplace=True)
impDF.sort_values("importances",ascending=False, inplace=True)
impDF.reset_index(drop=True, inplace=True)
resultDF = resultDF.fillna(0)
resultDF_1 = resultDF[["OIB", STORAGE_NAME]]
df_out = resultDF_1 
outputfile = STORAGE_NAME + ".csv"
df_out.to_csv(outputfile, encoding="utf8", index=False,sep="|")
outputfile2 = STORAGE_NAME +"_importances.csv"
impDF.to_csv(outputfile2, encoding="utf8", index=False,sep="|")