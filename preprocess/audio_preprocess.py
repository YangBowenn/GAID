"""
extraction audio from video
"""
from moviepy import VideoFileClip
import concurrent.futures
import os
from tqdm import tqdm
from multiprocessing import Pool
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count

video_folder = "Datasets/LSMDC/videos"
audio_folder = "Datasets/LSMDC/all_videos_audios"

def find_all_files(folder, extensions=None):
    if extensions is None:
        extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".3g2", '.3gp', '.qt', '.3gpp', '.mpg', '.m4v']
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, file))
    return all_files

def extract_audio_to_file(param):
    video_path, audio_path = param
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if not audio_clip:
            return False
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        audio_clip.close()
        return True
    except Exception as e:
        print(e)
        return False

all_videos = find_all_files(video_folder)
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

params = []
video_list, audio_list = [], []
for video_path in tqdm(all_videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
  
    audio_path = os.path.join(audio_folder, video_name+'.wav')
    if os.path.exists(audio_path):
        continue
    param = (video_path, audio_path)
    params.append(param)


num_works = cpu_count()
pool = Pool(num_works)
tqdm(pool.map(extract_audio_to_file, params))
pool.close()
pool.join()

