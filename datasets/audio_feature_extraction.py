import torch
import torchaudio
import whisper
from tqdm import tqdm
import os
import time

def load_audio(filepath: str, sample_rate: int=16000):
    waveform, original_sr = torchaudio.load(filepath)
    waveform_mono = waveform.mean(dim=0)
    if original_sr != sample_rate:
        waveform_mono = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq = sample_rate)(waveform_mono)
    return waveform_mono.squeeze(0).numpy()

def extract_features(audio: torch.Tensor, model: whisper.Whisper):
    """
    extract features with openai-whisper model
    """
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    # align length audio to whisper
    max_time_steps = model.encoder.positional_embedding.shape[0]
    if mel.shape[1] > max_time_steps*2:
        mel = mel[:, :max_time_steps*2]
    elif mel.shape[1] < max_time_steps*2:
        padding = max_time_steps*2 - mel.shape[1]
        mel = torch.nn.functional.pad(mel, (0, padding))
    mel = mel.unsqueeze(0)
    with torch.no_grad():
        features = model.encoder(mel)
    return features
    
if __name__ == "__main__":
    model = whisper.load_model('base', download_root="pretrained/whisper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    last_count = 0
    while True:
        audio_folder = "Datasets/LSMDC/all_videos_audios"
        audio_feature_folder = "Datasets/LSMDC/all_videos_audios_feature"

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
            if os.path.exists(os.path.join(audio_feature_folder, audio_id+'.pt')):
                continue
            try:
                audio = load_audio(audio_path)
                features = extract_features(torch.tensor(audio).to(device), model)
                torch.save(features.cpu(), os.path.join(audio_feature_folder, audio_id+'.pt'))
            except Exception as e:
                print(e)
                print(audio_path)
                continue
        time.sleep(60)