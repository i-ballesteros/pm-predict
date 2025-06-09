from pathlib import Path
from google.cloud import storage


def download_gcs_prefix(
    client: storage.Client,
    gcs_path: str,
    dest_root: Path = None,
) -> Path:
    """
    Downloads all files under a GCS prefix (gs://bucket/prefix)
    into dest_root/prefix, preserving sub-directories.
    Returns the local Path to the downloaded folder.
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_path!r}")

    bucket_name, prefix = gcs_path[len("gs://") :].split("/", 1)
    bucket = client.bucket(bucket_name)

    if dest_root is None:
        dest_root = Path.cwd() / "data"
    local_base = dest_root / prefix
    local_base.mkdir(parents=True, exist_ok=True)

    for blob in bucket.list_blobs(prefix=prefix):
        # Skip dirs
        if blob.name.endswith("/"):
            continue
        # Compute the path relative to the prefix
        rel_path = Path(blob.name[len(prefix) :].lstrip("/"))
        local_path = local_base / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    return local_base
