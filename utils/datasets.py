import os
from pathlib import Path
from urllib.parse import urlparse


def get_dataset_filename(env):
    dataset_url = getattr(env.unwrapped, "dataset_url", None)
    if not dataset_url:
        return None
    return Path(urlparse(dataset_url).path).name


def ensure_dsrl_dataset(
    env,
    download="auto",
    repo_id="YYY-45/DSRL",
    endpoint="https://hf-mirror.com",
    local_dir=None,
):
    if download == "off":
        return None

    filename = get_dataset_filename(env)
    if not filename:
        return None

    dataset_dir = Path(local_dir or Path.home() / ".dsrl" / "datasets").expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_path = dataset_dir / filename
    if local_path.exists():
        print(f"Using local DSRL dataset: {local_path}")
        return local_path

    if download not in ("auto", "hf"):
        raise ValueError(f"Unsupported dataset download mode: {download}")

    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic DSRL dataset download. "
            "Install it with `pip install huggingface_hub` or set `--dataset-download off`."
        ) from exc

    print(f"Downloading DSRL dataset {filename} from Hugging Face repo {repo_id} to {dataset_dir}")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=dataset_dir,
        )
    )
