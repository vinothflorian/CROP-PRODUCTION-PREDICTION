# CROP-PRODUCTION-PREDICTION
 1. Data Cleaning and Preprocessing
Handle missing data and standardize column metrics.
Filter relevant columns for analysis data. 
2. Exploratory Data Analysis (EDA)
Analyze Crop Distribution
Crop Types: Study the distribution of the Item column to identify the most and least cultivated crops across regions.
Geographical Distribution: Explore the Area column to understand which regions focus on specific crops or have high agricultural activity.
Temporal Analysis
Yearly Trends: Analyze the Year column to detect trends in Area harvested, Yield, and Production over time.
Growth Analysis: Investigate if certain crops or regions show increasing or decreasing trends in yield or production.
Environmental Relationships
Although explicit environmental data is absent, infer relationships between the Area harvested and Yield to check if there’s an impact of resource utilization on crop productivity.
Input-Output Relationships
Study correlations between Area harvested, Yield, and Production to understand the relationship between land usage and productivity.
Comparative Analysis
Across Crops: Compare yields (Yield) of different crops (Item) to identify high-yield vs. low-yield crops.
Across Regions: Compare production (Production) across different areas (Area) to find highly productive regions.
Productivity Analysis
Examine variations in Yield to identify efficient crops and regions.
Calculate productivity ratios: Production/Area harvested to cross-verify yields.

3.Libraries used

 1. pandas as pd
 2.streamlit as st
 3.sklearn.linear_model import LinearRegression
 4.sklearn.tree import DecisionTreeRegressor
 5.sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
 6.sklearn.model_selection import train_test_split
 7.sklearn.preprocessing import LabelEncoder
 8.sklearn.metrics import r2_score
 9.numpy as np


4.Machine learning models used for prediction

 1.Linear Regression
 2.Decision Tree regressor
 3.Random Forest Regressor
 4. AdaBoost Regressor
