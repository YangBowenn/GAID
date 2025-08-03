import os
import torch
from torch.utils.data import Dataset
from config.base_config import Config
from modules.basic_utils import load_json
from datasets.video_capture import VideoCapture
import json

class VATEXDataset(Dataset):

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir  # data/VATEX/videos 
        self.audios_dir = config.audios_dir  # data/VATEX/audios
        self.img_transforms = img_transforms
        self.split_type = split_type
        pth = 'Datasets/vatex/new_anno/'
        
        self.clip2caption = []
        if split_type == 'train':
            train_file = os.path.join(pth, 'vatex_training_v1.0_subset.json')
            self.train_db = load_json(train_file)
            self.all_train_pairs = self._construct_all_pairs(self.train_db)
               
        else:
            test_file = os.path.join(pth, 'vatex_validation_v1.0_subset_split.json')
            self.test_db = load_json(test_file)
            self.all_test_pairs = self._construct_all_pairs(self.test_db, split_type)
        
        
    def _construct_all_pairs(self, db, split_type=None):
        all_pairs = []
        for item in db:
            video_id = item['videoID'][:11]
            captions = item['enCap']
            if split_type == 'test':
                all_pairs.append([video_id, captions[0], 0])
            else:
                for senid, caption in enumerate(captions):
                    all_pairs.append([video_id, caption, senid])
        return all_pairs


    def __getitem__(self, index):
        
        video_path, audio_path, caption, vid, senid = self._get_vidpath_and_caption_by_index(index)
        
        if os.path.exists(audio_path):
            audio = torch.load(audio_path)
        else:
            audio = torch.zeros(1, 1500, 512)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                        self.config.num_frames, 
                                                        self.config.video_sample_type)
        
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': vid,
            'video': imgs,
            'audio': audio,
            'text': caption,
        }

    
    def __len__(self):
        if self.split_type == "train":
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)


    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == "train":
            vid, caption, senid = self.all_train_pairs[index]
        else:
            vid, caption, senid = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        audio_path = os.path.join(self.audios_dir, vid + '.pt')
        return video_path, audio_path, caption, vid, senid

