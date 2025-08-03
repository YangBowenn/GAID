from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
import os
from tqdm import tqdm
import torch.nn.functional as F
import time

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("pretrained/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("pretrained/wav2vec2-base")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# audio preprocess
def preprocess_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)  # single channel
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.to(device)

# feature extraction
def extract_wav2vec2_feature(file_path, target_frames=1500):
    waveform = preprocess_audio(file_path)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # shape: [1, T, 768]

    features = hidden_states.permute(0, 2, 1)
    
    resized = F.interpolate(features, size=target_frames, mode="linear", align_corners=True)
    
    final = resized.permute(0, 2, 1).cpu()
    return final


    
if __name__ == "__main__":
    
    last_count = 0
    while True:
        audio_folder = "Datasets/msrvtt/MSRVTT/audios"
        audio_feature_folder = "data/MSRVTT_audio_features_wav"

        if not os.path.exists(audio_feature_folder):
            os.makedirs(audio_feature_folder)
        all_videos = os.listdir(audio_folder)
        current_count = len(all_videos)
        if current_count > last_count:
            last_count = current_count
        else:
            break
        for audio_path in tqdm(all_videos):
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]
            audio_path = os.path.join(audio_folder, audio_path)

            try:
                features = extract_wav2vec2_feature(audio_path)
                torch.save(features, os.path.join(audio_feature_folder, audio_id+'.pt'))
            except Exception as e:
                print(e)
                print(audio_path)
                continue
        time.sleep(60)