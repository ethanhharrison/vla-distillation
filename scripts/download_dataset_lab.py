"""Same as download_dataset.py, for this shared lab machine.

`download_dataset.py` works as committed on a personal machine authenticated
via `gcloud auth login` (a real user session with a default project). This
machine only has Application Default Credentials for a service-style login
(`gcloud auth application-default login`), which has no project embedded -
and the two further bugs that surfaces don't show up on a normal login:

- `google.cloud.storage.Client()` raises "Project was not passed and could
  not be determined from the environment" without one.
- Passing a project explicitly to work around that then attaches a
  `userProject` to every request, which this (non-requester-pays) bucket
  rejects with "User project specified in the request is invalid."
- The default bucket, `pranav-us-east5`, 403s for us regardless - our access
  is to `pranav-us-central1` (see ~/proj/notes/vla/HANDOFF.md).

Fixed here with plain authenticated REST calls (no project needed at all)
instead of the `google-cloud-storage` client, and the working bucket as the
default. `download_dataset.py` itself is left untouched so it keeps working
wherever it already does.

Usage: same as download_dataset.py.
    python scripts/download_dataset_lab.py
"""

import random
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _access_token() -> str:
    """A fresh OAuth token from Application Default Credentials."""
    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def download_cs_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from a GCS bucket."""
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{source_blob_name.replace('/', '%2F')}"
    headers = {"Authorization": f"Bearer {_access_token()}"}
    with requests.get(url, headers=headers, params={"alt": "media"}, stream=True) as resp:
        resp.raise_for_status()
        with open(destination_file_name, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    print(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_name}.")


def list_gcs_tfrecords(bucket_name, folder_prefix):
    """Lists all tfrecords in a GCS folder, including subfolders."""
    headers = {"Authorization": f"Bearer {_access_token()}"}
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
    file_names = []
    params = {"prefix": folder_prefix}
    while True:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        file_names += [item["name"] for item in data.get("items", []) if item["name"].endswith(".tfrecord")]
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return file_names

def download_droid_records(
    num_records,
    bucket_name="pranav-us-central1",
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
