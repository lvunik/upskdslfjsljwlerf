import os
import shutil
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

def to_base36(num):
    if num == 0:
        return '0'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    res = ''
    while num:
        num, i = divmod(num, 36)
        res = alphabet[i] + res
    return res

def generate_id(movie_id_str):
    try:
        num = int(movie_id_str)
        return to_base36(num * 987654321 + 12345)
    except ValueError:
        return str(movie_id_str)

def upload_and_cleanup(output_dir, movie_id):
    hf_username = os.environ.get('HF_USERNAME')
    if not hf_username:
        print("HF_USERNAME environment variable not set. Skipping Hugging Face upload.")
        return
    
    encoded_movie_id = generate_id(movie_id)
    repo_id = f"{hf_username}/{encoded_movie_id}"
    
    if HfApi is None:
        print("huggingface_hub is not installed. Please run `pip install huggingface_hub`. Skipping upload.")
        return

    print(f"Uploading {movie_id} to Hugging Face Repo: {repo_id}...")
    api = HfApi()
    
    try:
        # Create the repository if it doesn't already exist
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        
        api.upload_large_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=[f"{movie_id}.m3u8"]
        )
        print(f"Successfully uploaded {movie_id} to Hugging Face.")
    except Exception as e:
        print(f"Failed to upload {movie_id} to Hugging Face: {e}")
        return

    # Cleanup local folder
    try:
        shutil.rmtree(output_dir)
        print(f"Successfully deleted local directory {output_dir}")
    except Exception as e:
        print(f"Failed to delete local directory {output_dir}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python upload.py <movie_directory_path>")
        print("Example: python upload.py movie/795390")
        sys.exit(1)
        
    dir_path = sys.argv[1].rstrip('/')
    if os.path.exists(dir_path):
        movie_id = os.path.basename(os.path.abspath(dir_path))
        upload_and_cleanup(dir_path, movie_id)
    else:
        print(f"Directory {dir_path} does not exist.")
