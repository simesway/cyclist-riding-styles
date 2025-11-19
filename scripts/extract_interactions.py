from data.load import load_all_trajectories, load_metadata
from maneuvers.interaction_detection import get_interactions, save_interactions_to_csv

df = load_all_trajectories("../data/raw")

metadata = load_metadata("../data/raw/meta_information.json")
labels = metadata["categories"]

# Store bicycle-bicycle interactions

bicycle_df = df[df["category"] == labels["bicycle"]]

bicycle_ids = bicycle_df["track_id"].unique()

distances = [10, 25, 50]
for d in distances:
  interactions = get_interactions(bicycle_df, bicycle_ids, distance=d, batch_size=10, min_duration=0)
  save_interactions_to_csv(interactions, f"../data/events/interactions/interactions_cyclist-cyclist_radius_{d}m.csv")