import random
from pathlib import Path

from google.cloud import storage

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def download_cs_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_name}.")

def list_gcs_tfrecords(bucket_name, folder_prefix):
    """Lists all tfrecords in a GCS folder, including subfolders."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_prefix)
    file_names = [b.name for b in blobs if b.name.endswith('.tfrecord')]
    return file_names

def download_droid_records(
    num_records,
    bucket_name="pranav-us-east5", 
    droid_folder="datasets/droid/success/", 
    shuffle=True
):
    record_names = list_gcs_tfrecords(bucket_name, droid_folder)
    if shuffle:
        random.shuffle(record_names)
    selected_records = record_names[:num_records]
    for record_name in selected_records:
        destination = PROJECT_ROOT / record_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_cs_file(bucket_name, record_name, str(destination))

if __name__ == "__main__":
    download_droid_records(3)