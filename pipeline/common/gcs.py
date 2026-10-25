"""The one thing callers need to know about GCS paths: don't mangle them.

TensorFlow already reads `gs://` paths natively (`tf.data.TFRecordDataset`,
`tf.io.gfile`) using the same Application Default Credentials already set up
on this machine - confirmed working against `pranav-us-central1` with zero
extra auth code. So there is no custom download/streaming layer here.

The one real hazard: `pathlib.Path("gs://bucket/x")` collapses the double
slash to `gs:/bucket/x`, silently corrupting the URI. Any code that takes a
record path that might be a `gs://` URI should keep it as a plain string
(only use `Path(...)` for genuinely-local operations like `.stem`, which is
unaffected by the collapse) rather than round-tripping it through `Path`.
`is_gcs_path` is the one predicate for code that needs to branch on it.

Uploading is not implemented: this account currently has read-only access to
`pranav-us-central1` (confirmed with a live write attempt - 403
storage.objects.create denied), so there is nothing to build yet.
"""

from __future__ import annotations


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")
