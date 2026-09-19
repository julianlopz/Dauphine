import polars as pl

df = pl.read_csv("data/output/2.clean_df.csv")

df = df.rename({
    "Start date": "start_date",
    "Start station number": "start_station",
    "End date": "end_date",
    "End station number": "end_station",
    "Total duration (ms)": "duration_ms",
    "Operating date": "operating_date",
    "Day of week": "day_of_week",
    "Hour of day": "hour"
})

df = df.with_columns([
    pl.col("start_date").str.to_datetime(strict=False),
    pl.col("end_date").str.to_datetime(strict=False),
    pl.col("operating_date").str.to_date(strict=False),
])

#########################
#Create station profiles
#########################

#-------------------
#departure features
#-------------------
departures = (
    df
    .group_by("start_station")
    .agg([
        pl.len().alias("n_departures"),

        pl.col("duration_ms").mean().alias("mean_duration"),
        pl.col("duration_ms").median().alias("median_duration"),
        pl.col("duration_ms").std().alias("std_duration"),

        pl.col("end_station").n_unique().alias("n_destinations"),
    ])
    .rename({"start_station": "station"})
)


#--------------------
#arrival features
#--------------------
arrivals = (
    df
    .group_by("end_station")
    .agg([
        pl.len().alias("n_arrivals"),

        pl.col("start_station").n_unique().alias("n_origins"),
    ])
    .rename({"end_station": "station"})
)


#Join them
stations = departures.join(
    arrivals,
    on="station",
    how="full"
).fill_null(0)



#----------------
#add departures per hour
#----------------

hour_profile = (
    df
    .group_by(["start_station", "hour"])
    .agg(
        pl.len().alias("count")
    )
)


hour_profile = (
    hour_profile
    .pivot(
        on="hour",
        index="start_station",
        values="count"
    )
    .fill_null(0)
)


hour_profile = hour_profile.rename({
    "start_station": "station"
})


#--------------------
# Normalize hourly profile
#--------------------

hour_columns = [str(i) for i in range(24) if str(i) in hour_profile.columns]

hour_profile = hour_profile.with_columns(
    [
        (pl.col(c) / pl.sum_horizontal(hour_columns)).alias(f"{c}_hours")
        for c in hour_columns
    ]
).drop(hour_columns)


#-----------------------
#Weekend and weekday behavior
#-----------------------
print(
    df
    .select("day_of_week")
    .unique()
    .sort("day_of_week")
)

df = df.with_columns(
    (pl.col("day_of_week") >= 5).alias("is_weekend")
)

weekend_profile = (
    df
    .group_by("start_station")
    .agg([
        pl.col("is_weekend").mean().alias("weekend_ratio")
    ])
    .rename({"start_station": "station"})
)


#--------------------
#Rush hour features
#--------------------

df = df.with_columns([
    (
        (pl.col("hour") >= 7) &
        (pl.col("hour") <= 9)
    ).alias("morning_rush"),

    (
        (pl.col("hour") >= 17) &
        (pl.col("hour") <= 19)
    ).alias("evening_rush"),

    (
        (pl.col("hour") >= 0) &
        (pl.col("hour") <= 5)
    ).alias("night")
])

rush_profile = (
    df
    .group_by("start_station")
    .agg([
        pl.col("morning_rush").mean().alias("morning_rush_ratio"),
        pl.col("evening_rush").mean().alias("evening_rush_ratio"),
        pl.col("night").mean().alias("night_ratio"),
    ])
    .rename({"start_station": "station"})
)


#-------------------
#Final df -> stations
#-------------------

stations = (
    stations
    .join(hour_profile, on="station", how="left")
    .join(weekend_profile, on="station", how="left")
    .join(rush_profile, on="station", how="left")
    .fill_null(0)
)

print(stations)
print(stations.shape)


#######################
#export df
#######################

stations.write_csv("data/output/4.stations_df.csv")

