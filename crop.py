import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


tab1, tab2 = st.tabs(["Home Page","Insights"])

df = pd.read_csv(r"D:\Python Projects\Production\CROP PRODUCTION\FAOSTAT_data.csv")
df.drop(['Domain Code', 'Note', 'Flag Description', 'Area', 'Domain', 'Year Code','Unit'], axis=1, inplace = True)
df['Value'].fillna(0,inplace = True)
df = df.dropna(subset= ['Flag'], axis=0)
#df = df.dropna(subset= ['Unit'], axis=0)
df['Value'] = df['Value'].astype(float)

df['Unique_Code'] =  df['Area Code (M49)'].astype(str)+"-"+ df['Item Code (CPC)'].astype(str)+"-"+df['Year'].astype(str)

# Separate data frame for area harvested, yield and production
DF_AH = df[df['Element']== 'Area harvested'].reset_index()
DF_YL = df[df['Element']== 'Yield'].reset_index()
DF_PRD = df[df['Element']== 'Production'].reset_index()

# CONCATINATE THE DATAFRAMES
DF_AHYL = DF_AH.merge( DF_YL, on= 'Unique_Code', how = 'inner')

DF_FINAL = DF_PRD.merge(DF_AHYL, on= 'Unique_Code', how = 'inner')

#DF_FINAL.drop([], axis= 1)

DF_FINAL.drop(['index_x', 'Area Code (M49)_x',
       'Element Code_x', 'Element_x', 'Item_x', 'Year_x',  'Flag_x',
       'index_y', 'Area Code (M49)_y', 'Element Code_y', 'Element_y', 'Item_y',
       'Year_y',  'Flag_y', 'Unique_Code', 'index', 'Element', 'Element Code', 'Item Code (CPC)_x', 'Item Code (CPC)_y', 'Item', 'Flag' ], axis =1, inplace=True)




DF_FINAL = DF_FINAL.rename(columns={'Value_x': 'Area Harvesting', 'Value_y': 'Yeild', 'Value': 'Production'})

encoder = LabelEncoder()

#DF_FINAL['Flag'] = encoder.fit_transform(DF_FINAL['Flag'])
#DF_FINAL = DF_FINAL.dropna(subset= ['Item Code (CPC)'], axis=0)

df['Value'] = df['Value'].astype(float)

DF_FINAL = DF_FINAL.reindex(columns=['Area Code (M49)', 'Item Code (CPC)', 'Year', 'Area Harvesting', 'Yeild', 'Production'])

DF_FINAL['Area Code (M49)'] = DF_FINAL['Area Code (M49)'].astype(float)
DF_FINAL['Item Code (CPC)'] = DF_FINAL['Item Code (CPC)'].astype(float)
DF_FINAL['Year'] = DF_FINAL['Year'].astype(float)
DF_FINAL['Area Harvesting'] = DF_FINAL['Area Harvesting'].astype(float)
DF_FINAL['Yeild'] = DF_FINAL['Yeild'].astype(float)



models = [LinearRegression(),DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor()]
r2_Score = []

x = DF_FINAL.drop(['Production'], axis=1)
y = DF_FINAL['Production']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
for model in models:
        model.fit(x_train,y_train)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        r2_Score.append(r2_score(y_test,test_prediction))

print(r2_Score)
best_r2score = max(r2_Score)
best_modelname =  r2_Score.index(best_r2score)

print (f"the best modef is: {models[best_modelname]}")

model1 = models[best_modelname]

with tab1:
        st.write(x_train)