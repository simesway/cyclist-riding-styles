from data.load import load_all_trajectories, load_metadata
from maneuvers.interaction_detection import get_interactions, save_interactions_to_csv

extract_bicycle_interactions = True
extract_car_interactions = True
extract_pedestrian_interactions = True

print("loading trajectories...")
df = load_all_trajectories("../data/raw")

metadata = load_metadata("../data/raw/meta_information.json")
labels = metadata["categories"]

distances = [5, 10, 25]

# Store bicycle-bicycle interactions

bicycle_df = df[df["category"] == labels["bicycle"]]
bicycle_ids = bicycle_df["track_id"].unique()

if extract_bicycle_interactions:
  print("extracting bicycle-bicycle interactions")
  for d in distances:
    path = f"../data/events/interactions/interactions-cyclist-cyclist-{d}m.csv"
    interactions = get_interactions(
      bicycle_df,
      bicycle_ids,
      distance=d,
      batch_size=10,
      min_duration=0,
      show_progress=True,
      exclude_ids=[]
    )
    save_interactions_to_csv(interactions, path)


# Store bicycle-car interactions

car_df = df[df["category"].isin([labels["car"], labels["bicycle"]])]

if extract_car_interactions:
  print("extracting bicycle-car interactions")
  for d in distances:
    path = f"../data/events/interactions/interactions-cyclist-car-{d}m.csv"
    interactions = get_interactions(
      car_df,
      bicycle_ids,
      distance=d,
      batch_size=10,
      min_duration=0,
      show_progress=True,
      exclude_ids=bicycle_ids
    )
    save_interactions_to_csv(interactions, path)


# Store bicycle-pedestrian interactions

pedestrian_df = df[df["category"].isin([labels["pedestrian"], labels["bicycle"]])]

if extract_pedestrian_interactions:
  print("extracting bicycle-pedestrian interactions")
  for d in distances:
    path = f"../data/events/interactions/interactions-cyclist-pedestrian-{d}m.csv"
    interactions = get_interactions(
      pedestrian_df,
      bicycle_ids,
      distance=d,
      batch_size=10,
      min_duration=0,
      show_progress=True,
      exclude_ids=bicycle_ids
    )
    save_interactions_to_csv(interactions, path)