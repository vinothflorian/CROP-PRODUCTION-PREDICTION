import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import numpy as np


tab1, tab2 = st.tabs(["Home Page","Insights"])

df_source = pd.read_csv(r"D:\Python Projects\Production\CROP PRODUCTION\FAOSTAT_data.csv")

#region Data Cleaning
df = pd.read_csv(r"D:\Python Projects\Production\CROP PRODUCTION\FAOSTAT_data.csv")
df.drop(['Domain Code', 'Note', 'Flag Description', 'Area', 'Domain', 'Year Code','Unit'], axis=1, inplace = True)
df['Value'].fillna(0,inplace = True)
df = df.dropna(subset= ['Flag'], axis=0)
#df = df.dropna(subset= ['Unit'], axis=0)
df['Value'] = df['Value'].astype(float)
#endregion

#region Data Preprocessing
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
#endregion

#region Machine Learning Algorithm
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



model1 = models[best_modelname]
#endregion

#region Stremlit Tab1 for Prediction
with tab1:
        st.title("Predicting Crop Production Based on Agricultural Data")
        st.divider() 
        st.header("Algorithm used to Predict", divider="green" )
        st.subheader (f"Best Model: :blue[{models[best_modelname]}]")
        st.subheader (f"R2 Score: :blue[{max(r2_Score)}] ")
        st.divider() 
        st.header("DataPoints For Prediction", divider= "green")
        Ar_Code = st.number_input("Enter Area Code")
        Item_code = st.number_input("Enter Item Code")
        Year = st.number_input("Enter Year")
        Area_Harvesting = st.number_input("Enter Area Harvested")
        Yeild = st.number_input("Enter Yeild")
        NEW_DF = pd.DataFrame()
        NEW_DF['Area Code (M49)'] = [Ar_Code]
        NEW_DF['Item Code (CPC)'] = [Item_code]
        NEW_DF['Year'] = [Year]
        NEW_DF['Area Harvesting'] = [Area_Harvesting]
        NEW_DF['Yeild'] = [Yeild]

        if st.button('Prediction'):
                NEW_DF['Area Code (M49)'] = NEW_DF['Area Code (M49)'].astype(float)
                NEW_DF['Item Code (CPC)'] = NEW_DF['Item Code (CPC)'].astype(float)
                NEW_DF['Year'] = NEW_DF['Year'].astype(float)
                NEW_DF['Area Harvesting'] = NEW_DF['Area Harvesting'].astype(float)
                NEW_DF['Yeild'] = NEW_DF['Yeild'].astype(float)
                Prediction = model1.predict(NEW_DF)
                st.write(f"The Predicted Production Value : :red{Prediction}")

#endregion

#region DATAFRAME FOR EDA

DF_AREA = df_source[['Area Code (M49)', 'Area']]
DF_AREA = DF_AREA.drop_duplicates()
DF_ITEM = df_source[['Item Code (CPC)', 'Item']]
DF_ITEM = DF_ITEM.drop_duplicates()
df['Item Code (CPC)'] = pd.to_numeric(df['Item Code (CPC)'], errors='coerce')
DF_ITEM.dropna(axis=0, inplace=True)
DF_ITEM = DF_ITEM[DF_ITEM['Item Code (CPC)'] != '2351f']
DF_ITEM['Item Code (CPC)'] = DF_ITEM['Item Code (CPC)'].astype(float)


DF_EDA = pd.merge(DF_FINAL, DF_AREA, how='inner', left_on= 'Area Code (M49)', right_on= 'Area Code (M49)')
DF_EDA = pd.merge(DF_EDA, DF_ITEM, how='inner', left_on= 'Item Code (CPC)', right_on= 'Item Code (CPC)')

DF_EDA['Year'] = np.floor(DF_EDA['Year'])

DF_EDA.loc[DF_EDA['Area'] == 'China, Macao SAR', 'Area'] = 'China'
DF_EDA.loc[DF_EDA['Area'] == 'China, mainland', 'Area'] = 'China'
DF_EDA['Productivity Ratio'] = DF_EDA['Production']/DF_EDA['Area Harvesting']
print(DF_EDA)

#endregion

#region streamlit Tab2 for EDA
with tab2:

        options = st.selectbox("Select The Data Analysis", ('Analyze Crop Distribution', 'Temporal Analysis', 
                                                            'Input-Output Relationships', 'Comparative Analysis', 'Productivity Analysis'))
        
        if options == 'Analyze Crop Distribution':
                st.header("Area Wise Crop Distribution", divider='green')
                pivot_areadf = DF_EDA.pivot_table(index='Area', values= 'Item', aggfunc= 'count')
                top5_pivot = pivot_areadf.sort_values(by='Item', ascending=False)
                top5 = top5_pivot.head(5)
                ls5 = top5_pivot.tail(5)
                st.write(pivot_areadf)
                st.write("Top 5 Crop Producing countries")
                st.bar_chart(top5)
                st.write("Least 5 Crop Producing countries")
                st.bar_chart(ls5)
        elif options == 'Temporal Analysis':
                st.header("Crop Production - Yearly Trends", divider='green')
                pivot_yeardf = DF_EDA.pivot_table(index='Year', values= 'Production', aggfunc= 'sum')
                st.line_chart(pivot_yeardf)
        elif options == 'Input-Output Relationships':
                st.header("Input-Output Relationships", divider='green')
                pivot_yeardf = DF_EDA.pivot_table(index='Year', values= ['Area Harvesting','Yeild','Production'], aggfunc= 'sum')
                st.bar_chart(pivot_yeardf)
        elif options == 'Comparative Analysis':
                st.header("Comparative Analysis", divider='green')
                pivot_yeilddf = DF_EDA.pivot_table(index='Item', values= 'Yeild', aggfunc= 'sum')
                top5_ylpivot = pivot_yeilddf.sort_values(by='Yeild', ascending=False)
                top5yl = top5_ylpivot.head(5)
                ls5yl = top5_ylpivot.tail(5)
                st.write("Top 5 Yeild")
                st.bar_chart(top5yl)
                st.write("Bottom 5 Yeild")
                st.bar_chart(ls5yl)
        elif options == 'Productivity Analysis':
                st.header("Productivity Analysis", divider='green')
                pivot_yeilddf = DF_EDA.pivot_table(index='Area', values= 'Yeild', aggfunc= 'sum')
                top5_ylpivot = pivot_yeilddf.sort_values(by='Yeild', ascending=False)
                top5yl = top5_ylpivot.head(5)
                st.write("Top 5 Efficient Region by Yeild")
                st.bar_chart(top5yl)
                DF_PR = DF_EDA[['Area', 'Item','Yeild', 'Productivity Ratio']]
                st.write("Area with Item wise Productivity Ratio")
                st.write(DF_PR)
#endregion
                


                


        




        
