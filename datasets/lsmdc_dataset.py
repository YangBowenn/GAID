import os
import torch
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class LSMDCDataset(Dataset):

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.audios_dir = config.audios_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        
        self.clip2caption = {}
        if split_type == 'train':
            pth = 'Datasets/LSMDC/'
            train_file = pth + 'LSMDC16_annos_training_someone.csv'
            self._compute_clip2caption(train_file)
               
        else:
            pth = 'Datasets/LSMDC/'
            test_file = pth + 'LSMDC16_challenge_1000_publictect.csv'
            self._compute_clip2caption(test_file)
  

    def __getitem__(self, index):
        video_path, caption, video_id, audio_path = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)
        if os.path.exists(audio_path):
            audio = torch.load(audio_path)
        else:
            audio = torch.zeros(1, 1500, 512)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'audio': audio,
            'text': caption,
        }

    
    def __len__(self):
        return len(self.clip2caption)

    def _get_vidpath_and_caption_by_index(self, index):
        clip_id = list(self.clip2caption.keys())[index]
        if '"' in clip_id:
            clip_id = clip_id[1:]
        caption = self.clip2caption[clip_id]
        clip_prefix = clip_id.split('.')[0][:-3]
        video_path = os.path.join(self.videos_dir, clip_id + '.avi')
        audio_path = os.path.join(self.audios_dir, clip_id + '.pt')
        return video_path, caption, clip_id, audio_path

            
    def _compute_clip2caption(self, csv_file):
        with open(csv_file, 'r', newline='', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, _, _, _, _, caption = line_split
                if '"' in clip_id:
                    clip_id = clip_id[1:]
                except_ls = ['1012_Unbreakable_00.05.16.065-00.05.21.941', '1012_Unbreakable_00.24.58.622-00.25.06.004'
                             ]
                if any(keyword in clip_id for keyword in except_ls):
                    continue
                self.clip2caption[clip_id] = caption
