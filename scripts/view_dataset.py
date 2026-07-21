import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def save_image(feature, name, output_dir):
    """Saves the first frame of a JPEG image feature to disk."""
    values = feature.bytes_list.value
    output_path = Path(output_dir) / f"{name.replace('/', '_')}.jpeg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(values[0])
    return output_path

def sample_value(feature):
    kind = feature.WhichOneof("kind")
    if kind == "int64_list":
        return list(feature.int64_list.value)
    if kind == "float_list":
        return list(feature.float_list.value)
    if kind == "bytes_list":
        values = feature.bytes_list.value
        if not values:
            return None
    return None

def get_features(
    dataset_folder=PROJECT_ROOT / "datasets/droid/success/",
    image_dir=PROJECT_ROOT / "images/",
):
    folder_path = Path(dataset_folder)
    filenames = [str(f) for f in folder_path.iterdir() if f.is_file()]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature
        for name in sorted(features):
            feature = features[name]
            kind = feature.WhichOneof("kind")
            print(f"  {name}, type={kind}")
            if "image" in name and kind == "bytes_list":
                output_path = save_image(feature, name, image_dir)
                print(f"    saved first frame to {output_path}")
            else:
                print(f"    sample={sample_value(feature)}")

if __name__ == "__main__":
    get_features()
