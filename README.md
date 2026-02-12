# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
# Step 1: Import Required Libraries

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Read the Dataset

# Replace with your actual CSV file
df1 = pd.read_csv('Data_set (1).csv')
df1.head()

# Step 3: Dataset Information
df1.info()
df1.describe()

# Step 4: Handling Missing Values
# Check Null Values
df1.isnull()
df1.isnull().sum()

# Fill Missing Values with 0
df1_fill_0 = df1.fillna(0)
df1_fill_0

# Forward Fill
df1_ffill = df1.ffill()
df1_ffill

# Backward Fill
df1_bfill = df1.bfill()
df1_bfill

# Fill with Mean (Numerical Column Example)
col = 'CoapplicantIncome'
if col in df1.columns:
    df1[col] = df1[col].fillna(df1[col].mean())
else:
    print(f"Column '{col}' not found. Available columns: {df1.columns.tolist()}")

# Show dataframe (or a subset) for inspection
print(df1.head())

# Drop Missing Values
df1_dropna = df1.dropna()
df1_dropna

#Step 5: Save Cleaned Data
df1_dropna.to_csv('clean_data.csv', index=False)

# OUTLIER DETECTION
# Step 6: IQR Method (Using Iris Dataset)
ir = pd.read_csv('iris.csv')
ir.head()
ir.info()
ir.describe()   

#Boxplot for Outlier Detection
sns.boxplot(x=ir['sepal_width'])
plt.show()

# Calculate IQR
Q1 = ir['sepal_width'].quantile(0.25)
Q3 = ir['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:", IQR)

# Detect Outliers
outliers_iqr = ir[
    (ir['sepal_width'] < (Q1 - 1.5 * IQR)) |
    (ir['sepal_width'] > (Q3 + 1.5 * IQR))
]
outliers_iqr

# Remove Outliers
ir_cleaned = ir[
    ~((ir['sepal_width'] < (Q1 - 1.5 * IQR)) |
      (ir['sepal_width'] > (Q3 + 1.5 * IQR)))
]
ir_cleaned

# Step 7: Z-Score Method

data = [1,12,15,18,21,24,27,30,33,36,39,42,45,48,51,
        54,57,60,63,66,69,72,75,78,81,84,87,90,93]

df_z = pd.DataFrame(data, columns=['values'])
df_z

# Calculate Z-Scores
z_scores = np.abs(stats.zscore(df_z))
z_scores

# Detect Outliers
threshold = 3
outliers_z = df_z[z_scores > threshold]
print("Outliers:")
outliers_z

# Remove Outliers
df_z_cleaned = df_z[z_scores <= threshold]
df_z_cleaned
```
<img width="934" height="850" alt="Screenshot 2026-02-12 083350" src="https://github.com/user-attachments/assets/bdda73ea-9f8f-440d-9a3f-e10a890b5e15" />

<img width="581" height="243" alt="Screenshot 2026-02-12 083436" src="https://github.com/user-attachments/assets/c16f28e8-42c2-4247-8f49-76a5e826e3ea" />

<img width="757" height="750" alt="Screenshot 2026-02-12 083449" src="https://github.com/user-attachments/assets/e58835bd-3f54-4d77-9ef7-f37e8b701843" />

# pdf file

[vertopal.com_dataset-merged.pdf](https://github.com/user-attachments/files/25249437/vertopal.com_dataset-merged.pdf)

# Result
Thus, the given data is cleaned and saved sucessfully.

