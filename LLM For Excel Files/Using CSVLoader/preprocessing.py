import pandas as pd
import os

#This code involves pre-processing the xls file and fixing unwanted issues

df = pd.read_excel('sample1.xls', index_col=0)

print(f"Before Pre-processing, dataframe sample:\n\n{df.head()}")

df["Date"] = pd.to_datetime(df["Date"],dayfirst = True)

df.rename(columns={"First Name": "first_name","Last Name":"last_name"},inplace=True)

print(f"\n\nPre-processing complete! Dataframe is now\n\n{df.head()}")

df.to_excel('updated_sample_file.xlsx')