import polars as pl

df = pl.read_csv("data/output/stations_df.csv")

print(df)