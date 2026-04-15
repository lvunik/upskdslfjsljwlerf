import os
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()

try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.hf_api import CommitOperationDelete
except ImportError:
    print("Please install huggingface_hub with `pip install huggingface_hub`")
    exit(1)

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

def process_movie(movie_id):
    hf_username = os.environ.get('HF_USERNAME')
    if not hf_username:
        print("HF_USERNAME is not set. Please set it in .env file.")
        return False
        
    encoded_movie_id = generate_id(movie_id)
    repo_id = f"{hf_username}/{encoded_movie_id}"
    
    api = HfApi()
    
    orig_dir = "./orig"
    movie_dir = "./movie"
    
    # 1. Clean up local directories if they exist
    if os.path.exists(orig_dir):
        shutil.rmtree(orig_dir)
    if os.path.exists(movie_dir):
        shutil.rmtree(movie_dir)
        
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(movie_dir, exist_ok=True)
    
    print(f"[{movie_id}] Downloading repo {repo_id} to {orig_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            local_dir=orig_dir,
            allow_patterns=["*.png", "*.m3u8"] # only download pngs and the index to save time (if there are other files, ignore)
        )
    except Exception as e:
        print(f"[{movie_id}] Failed to download repo {repo_id}: {e}")
        return False
        
    # 2. Rename .png to .ts
    print(f"[{movie_id}] Renaming .png files to .ts in {orig_dir}...")
    png_files = [f for f in os.listdir(orig_dir) if f.endswith('.png')]
    if not png_files:
        print(f"[{movie_id}] No .png files found in repo. Let's check maybe they are already .ts?")
        ts_count = len([f for f in os.listdir(orig_dir) if f.endswith('.ts')])
        if ts_count == 0:
            print(f"[{movie_id}] No segments found to process.")
            return False
            
    for f in png_files:
        base_name = f[:-4]
        os.rename(os.path.join(orig_dir, f), os.path.join(orig_dir, f"{base_name}.ts"))
        
    # 3. Run ffmpeg command
    ffmpeg_cmd = (
        'ffmpeg -i ./orig/index.m3u8 '
        '-c copy -hls_key_info_file ./enc.keyinfo '
        '-hls_flags independent_segments '
        '-hls_start_number 0 '
        '-hls_segment_filename "./movie/%03d.ts" '
        './movie/index.m3u8'
    )
    
    print(f"[{movie_id}] Running ffmpeg...")
    result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{movie_id}] ffmpeg failed:\n{result.stderr}")
        return False
        
    # User requested to keep the old index.m3u8 and NOT keep the new index.m3u8
    # ffmpeg generated ./movie/index.m3u8. We will delete it so it is NOT uploaded.
    # The original index.m3u8 on Hugging Face will therefore remain untouched.
    if os.path.exists("./movie/index.m3u8"):
        os.remove("./movie/index.m3u8")
        
    print(f"[{movie_id}] Overriding local existing original...")
    shutil.rmtree(orig_dir)
    os.rename(movie_dir, orig_dir)
        
    # 4. Upload back to Hugging Face
    print(f"[{movie_id}] Uploading new files back to {repo_id}...")
    try:
        api.upload_large_folder(
            folder_path=orig_dir,
            repo_id=repo_id,
            repo_type="dataset",
            # We explicitly ignore the old movie_id.m3u8 just in case, mimicking upload.py if relevant
            ignore_patterns=[f"{movie_id}.m3u8"]
        )
    except Exception as e:
        print(f"[{movie_id}] Failed upload: {e}")
        return False
        
    # 5. Delete old .png files from Hugging Face
    print(f"[{movie_id}] Deleting old .png files from repo...")
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        png_to_delete = [f for f in repo_files if f.endswith('.png')]
        if png_to_delete:
            operations = [CommitOperationDelete(path_in_repo=f) for f in png_to_delete]
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Delete unencrypted .png files",
                operations=operations
            )
            print(f"[{movie_id}] Deleted {len(png_to_delete)} .png files.")
    except Exception as e:
        print(f"[{movie_id}] Failed to delete old .png files: {e}")
        
    # 6. Cleanup locally
    print(f"[{movie_id}] Cleaning up local directories...")
    try:
        if os.path.exists(orig_dir):
            shutil.rmtree(orig_dir)
    except Exception as e:
        print(f"[{movie_id}] Local cleanup warning: {e}")
        
    print(f"[{movie_id}] Successfully finished processing.")
    return True

if __name__ == "__main__":
    input_file = "input.txt"
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(lines)} IDs to process in {input_file}.")
        for i, m_id in enumerate(lines, 1):
            print(f"\n--- Processing {i}/{len(lines)}: {m_id} ---")
            success = process_movie(m_id)
            if not success:
                print(f"Warning: Processing failed for {m_id}. Moving to next.")
    else:
        print(f"{input_file} not found. Please create it with movie IDs to process.")
