import polars as pl

df = pl.read_csv("data/output/2.clean_df.csv")

#-------------------------------
#le nombre total de trajets en 2025 ;
#--------------------------------

print(df.select(pl.count("Total duration (ms)")))

#After cleaning the data, there were 8'762,355 trips made in 2025

#-------------------------------
# le nombre ou le ratio de stations actives ;
#-------------------------------

#number of active start stations 
print(df.select(pl.col("Start station number").unique_counts()))

#820 start stations where used in 2025

#number of active end stations
print(df.select(pl.col("End station number").unique_counts()))

#822 end stations where used in 2025

#Note, this could change if the 2 stations that only appear in the end station number column are removed in the cleaning step.(station ids 10626 and 22168)

#Rate of active stations
start_ids = df["Start station number"].unique()
end_ids = df["End station number"].unique()

all_stations = pl.concat([start_ids, end_ids]).unique()

start_rate = start_ids.len() / all_stations.len()
end_rate = end_ids.len() / all_stations.len()

print(f"Start station active rate: {start_rate:.2%}")
print(f"End station active rate:   {end_rate:.2%}")

#almost 100% on both. which is normal actually since the empty stations where already filtered out

#Most used start and end station
print(df.group_by("Start station number").len().sort("len", descending=True))

print(df.group_by("End station number").len().sort("len", descending=True))




# -------------------------------
# le nombre ou le ratio de stations utilisées comme stations de départ ;
#-------------------------------

#check previous point


#-------------------------------
# le nombre ou le ratio de stations utilisées comme stations d'arrivée ;
#-------------------------------

#check previous point


#-------------------------------
# la distribution du nombre de trajets par jour ;
#-------------------------------

#Change date format to date time
#df = df.with_columns(
#    pl.col("Start date").str.to_datetime("%Y-%m-%d %H:%M"),
#    pl.col("End date").str.to_datetime("%Y-%m-%d %H:%M")
#)
#print(df)


print(df.group_by("Operating date").len().sort("len", descending=True))

#there is a huge gap between the 2 lowest dates: 2025-08-05 with 171 trajets and 2025-01-01 with 5077 trajets (could it be that the missing values removed in the cleaning belonged mostly to 2025-08-05 ?)


#-------------------------------
# la distribution du nombre de trajets par station ;
#-------------------------------

#Most used start and end station
print(df.group_by("Start station number").len().sort("len", descending=True))

print(df.group_by("End station number").len().sort("len", descending=True))

 
# -------------------------------
# la distribution des durées de trajets ;
#-------------------------------

print(df.group_by("Total duration (ms)").len().sort("len", descending=True))


#-------------------------------
# la répartition des trajets selon le jour de la semaine ;
#-------------------------------

#Day of week being 1=monday and 7=sunday
print(df.group_by("Day of week").len().sort("len", descending=True))

#lower use of the bicycles is seen on the weekend (6=saturday and 7=sunday)

#-------------------------------
# la répartition des trajets selon l'heure de la journée.
#-------------------------------

print(df.group_by("Hour of day").len().sort("len", descending=True))

#Most used hours are before work and after work. Least used are the night/morning hours (horas de madrugada - heures de nuit)


