import polars as pl

df = pl.read_csv("data/output/full_filtered_df.csv")

########################
#Missing start or end station number
########################
missing_values = df.null_count()

print(missing_values)

#184 missing values for columns end station number and total duration. if the missing values for the entries are not the same we have a total of 
# 184*2 = 368 out of 9068241 entries. Removing them would be a loss of less than 1% of the dataframe

df_no_nulls = df.drop_nulls()
print(df_no_nulls.shape)
print("---------------")



########################
#Total duration = 0, missing or negative
########################

df_duration = df_no_nulls.filter(pl.col("Total duration (ms)") <= 0)
print(df_duration)
print("-------------------")

# After removing the missing entries for the total duration, we see there are no negative or 0 ms entries 



########################
#same start and end station number
########################

#are there entries where the start and end stations are the same?
df_same_station = df_no_nulls.filter(pl.col("Start station number") == pl.col("End station number"))
print(df_same_station)

#305702 entries start and end in the same station. this represents 305702/9068057 = 0.033 -> 3.3% of the no_nulls dataset

#Lets filter them out
df_no_same_station = df_no_nulls.filter(pl.col("Start station number") != pl.col("End station number"))
print(df_no_same_station)

#new df shape is 8_762_355


########################
#incoherent start and end station numbers (?)
########################

#-------------------------
#Stations appearing in one column but not the other?
#-------------------------

#get uniqueids for both columns
start_ids = df_no_same_station["Start station number"].unique()
end_ids = df_no_same_station["End station number"].unique()

#compare them
start_only = start_ids.filter(~start_ids.is_in(end_ids))

end_only = end_ids.filter(~end_ids.is_in(start_ids))


print("Only in Start:", start_only)
print("Only in End:", end_only)

#There are 2 stations that only appear in the end station number column, station ids 10626 and 22168. should they be removed??

#-------------------------
# Negative station numbers?
#-------------------------

negative_stations = df_no_same_station.filter((pl.col("Start station number") < 0) | (pl.col("End station number") < 0))

print(negative_stations)

# No negative station number ids

#######################
# Add date columns
#######################
#Transform start and end date from text to date time
df = df_no_same_station.with_columns(
    pl.col("Start date").str.to_datetime("%Y-%m-%d %H:%M"),
    pl.col("End date").str.to_datetime("%Y-%m-%d %H:%M")
)

#Add Operating date, day of week and hour of day columns
df = df.with_columns(
    pl.col("Start date").dt.date().alias("Operating date"),
    pl.col("Start date").dt.weekday().alias("Day of week"),
    pl.col("Start date").dt.hour().alias("Hour of day"),
)

print(df)

#######################
#export df
#######################

df.write_csv("data/output/clean_df.csv")