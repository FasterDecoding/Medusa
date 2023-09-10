from huggingface_hub import HfApi
import argparse

parser = argparse.ArgumentParser("Upload Medusa model to HuggingFace Hub")
parser.add_argument("--folder", type=str, help="Path to model folder")
parser.add_argument("--repo", type=str, help="Repo name")
parser.add_argument("--private", action="store_true", help="Make repo private")

args = parser.parse_args()

api = HfApi()

api.create_repo(
    repo_id=args.repo,
    private=args.private,
    exist_ok=True,
)

api.upload_folder(
    folder_path=args.folder,
    repo_id=args.repo,
)