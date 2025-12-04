import pandas as pd

# close-ended
df_close = pd.read_csv("Data/mohler_close_ended.csv")
print(df_close.head())

# open-ended
df_open = pd.read_csv("Data/mohler_open_ended.csv")
print(df_open.head())

