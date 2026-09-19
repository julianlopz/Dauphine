import polars as pl
from pathlib import Path


files_path = Path("data/input_raw").glob("*.csv")

for file in Path("data/input_raw").glob("*.csv"):
    df = pl.read_csv(file)

    print(file)
    print(df.columns)
    print("----------------")

# the files have the same column names but not in the same order

################
# Read the files
################
files = sorted(Path("data/input_raw").glob("*.csv"))

# Use first file order
first_df = pl.read_csv(files[0])
columns = first_df.columns

#Read and reorder files
dfs = [pl.read_csv(file).select(columns)
       for file in files]

df = pl.concat(dfs)

print(df.head())
print(df.shape)
print("----------------------")
# All dfs are grouped into one giant df of shape (9068241, 11)


#########################
#Filter df to keep only useful columns
#######################


columns_to_drop = ["Number", "Start station", "End station", "Bike number", "Bike model", "Total duration"]

df_filtered = df.drop(columns_to_drop)

print(df_filtered.head())
print(df_filtered.columns)
print(df_filtered.shape)



###################
#Export csv
###################

df_filtered.write_csv("data/output/1.full_filtered_df.csv")