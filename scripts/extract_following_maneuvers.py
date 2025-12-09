import pandas as pd
from maneuvers.utils import save_maneuvers_to_csv
from maneuvers.following import get_following_maneuvers


interactions_df = pd.read_csv("../data/filtered_interactions-50m-no-stop.csv")
traj_df = pd.read_csv("../data/processed/bicycle_only.csv")

interactions_df = interactions_df[interactions_df["duration"] > 1]

config = {
    "min_length": 1.0,
    "max_lateral_distance": 2.0,
    "min_long_distance": 1.0,
    "max_long_distance": 35.0,
    "max_time_headway": 6.0,
    "max_rel_heading": 30,
}

maneuvers = get_following_maneuvers(traj_df, interactions_df, config)


save_maneuvers_to_csv(maneuvers, "../data/events/following/following1.csv")