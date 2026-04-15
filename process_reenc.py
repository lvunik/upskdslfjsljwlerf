import os
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    
    movie_dir = "./movie"
    temp_enc_dir = "./temp_enc"
    
    # 1. Clean up local directories if they exist
    for d in [movie_dir, temp_enc_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
            
    try:
        print(f"[{movie_id}] Downloading repo {repo_id} to {movie_dir}...")
        try:
            snapshot_download(
                repo_id=repo_id, 
                repo_type="dataset", 
                local_dir=movie_dir,
                allow_patterns=["*.png", "*.m3u8"], # only download pngs and the index to save time
                max_workers=4
            )
        except Exception as e:
            print(f"[{movie_id}] Failed to download repo {repo_id}: {e}")
            return False
            
        # 2. Rename .png to .ts
        print(f"[{movie_id}] Renaming .png files to .ts in {movie_dir}...")
        png_files = [f for f in os.listdir(movie_dir) if f.endswith('.png')]
        if not png_files:
            print(f"[{movie_id}] No .png files found in repo. Let's check maybe they are already .ts?")
            ts_count = len([f for f in os.listdir(movie_dir) if f.endswith('.ts')])
            if ts_count == 0:
                print(f"[{movie_id}] No segments found to process.")
                return False
                
        for f in png_files:
            base_name = f[:-4]
            os.rename(os.path.join(movie_dir, f), os.path.join(movie_dir, f"{base_name}.ts"))
            
        # 3. Run ffmpeg command
        ffmpeg_cmd = (
            f'ffmpeg -i {movie_dir}/index.m3u8 '
            f'-c copy -hls_key_info_file ./enc.keyinfo '
            f'-hls_flags independent_segments '
            f'-hls_start_number 0 '
            f'-hls_segment_filename "{temp_enc_dir}/%03d.ts" '
            f'{temp_enc_dir}/index.m3u8'
        )
        
        print(f"[{movie_id}] Running ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{movie_id}] ffmpeg failed:\n{result.stderr}")
            return False
            
        # User requested to keep the old index.m3u8 and NOT keep the new index.m3u8
        # ffmpeg generated {temp_enc_dir}/index.m3u8. We will delete it so it is NOT uploaded.
        if os.path.exists(f"{temp_enc_dir}/index.m3u8"):
            os.remove(f"{temp_enc_dir}/index.m3u8")
            
        print(f"[{movie_id}] Overwriting {movie_dir} with encrypted files...")
        shutil.rmtree(movie_dir)
        os.rename(temp_enc_dir, movie_dir)
            
        # 4. Upload back to Hugging Face
        print(f"[{movie_id}] Uploading new files back to {repo_id}...")
        try:
            api.upload_large_folder(
                folder_path=movie_dir,
                repo_id=repo_id,
                repo_type="dataset",
                # We explicitly ignore the old index.m3u8 just in case, mimicking upload.py if relevant
                num_workers=1
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
            
        print(f"[{movie_id}] Successfully finished processing.")
        return True

    finally:
        # 6. Cleanup locally in ALL cases
        print(f"[{movie_id}] Cleaning up local directories...")
        for d in [movie_dir, temp_enc_dir]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"[{movie_id}] Local cleanup warning for {d}: {e}")

if __name__ == "__main__":
    input_file = "input.txt"
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(lines)} IDs to process in {input_file}.")
        lines_to_process = list(lines)
        for i, m_id in enumerate(lines_to_process, 1):
            print(f"\n--- Processing {i}/{len(lines_to_process)}: {m_id} ---")
            success = process_movie(m_id)
            if success:
                lines.remove(m_id)
                with open(input_file, 'w') as f:
                    for remaining_m_id in lines:
                        f.write(f"{remaining_m_id}\n")
            else:
                print(f"Warning: Processing failed for {m_id}. Moving to next.")
    else:
        print(f"{input_file} not found. Please create it with movie IDs to process.")
