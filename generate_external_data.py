import fiftyone as fo
from fiftyone import ViewField as F
name="my-dataset"

dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    dataset_dir="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/val/",
    dataset_type=dataset_type,
    data_path="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/val/",
    labels_path="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/val/colorful_val.json",
    name=name,
)

session = fo.launch_app(dataset)
session.wait()
# view = (
#     dataset
#     .map_labels("detections", {"0":"box", "Box":"box", "good-parcel":"box", "Parcel":"box", "Box-broken":"box", "Boxes":"box",})
#     .filter_labels("detections", F("label") == "box")
# )

# colorful_dataset_train = view.clone()

# colorful_dataset_train.export(
#     dataset_dir="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/r_train/",
#     dataset_type=dataset_type,
#     data_path="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/r_train/",
#     labels_path="/home/lcy/Projects/BoxDetector/assets/colorful-dataset/r_train/colorful_train.json",
#     export_media="move",
# )

