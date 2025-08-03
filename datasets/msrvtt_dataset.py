import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
import random

class MSRVTTDataset(Dataset):

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.audios_dir = config.audios_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        dir = 'data/MSRVTT'
        db_file = dir + '/MSRVTT_data.json'
        test_file_pth = '/MSRVTT_JSFUSION_test.csv'
        test_csv = dir + test_file_pth

        if config.msrvtt_train_file == '7k':
            train_csv = dir + '/MSRVTT_train.7k.csv'
        else:
            train_csv = dir + '/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
    def __getitem__(self, index):

        if self.split_type == 'train':
            video_path, audio_path, caption, video_id, sen_id = self._get_vidpath_and_caption_by_index(index)
            
            if os.path.exists(audio_path):
                audio = torch.load(audio_path)
            else:
                if self.config.audio_encoder == "whisper":
                    audio = torch.zeros(1, 1500, 512)
                if self.config.audio_encoder == "wav2vec2":
                    audio = torch.zeros(1, 1500, 768)
                if self.config.audio_encoder == "ast":
                    audio = torch.zeros(1, 768)
                
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_id,
                'video': imgs,
                'audio': audio,
                'text': caption
            }
        else:
            video_path, audio_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)
            if os.path.exists(audio_path):
                audio = torch.load(audio_path)
            else:
                if self.config.audio_encoder == "whisper":
                    audio = torch.zeros(1, 1500, 512)
                if self.config.audio_encoder == "wav2vec2":
                    audio = torch.zeros(1, 1500, 768)
                if self.config.audio_encoder == "ast":
                    audio = torch.zeros(1, 768)
            # imgs, idxs = VideoCapture.load_frames_from_video_old(video_path,
            #                                                  self.config.num_frames,
            #                                                  self.config.video_sample_type)
            
            # sample_pos = self.video_dict[video_id]
            # imgs, idxs = VideoCapture.load_frames_from_video_keyframe(video_path, self.config.num_frames, sample_pos, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)


            return {
                'video_id': video_id,
                'video': imgs,
                'audio': audio,
                'text': caption,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption, senid = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            audio_path = os.path.join(self.audios_dir, vid + '.pt')
            # text_pos = random.choice([cap for cap, sid in zip(self.vid2caption[vid], self.vid2senid[vid]) if sid != senid])
            return video_path, audio_path, caption, vid, senid
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            audio_path = os.path.join(self.audios_dir, vid + '.pt')
            caption = self.test_df.iloc[index].sentence
            return video_path, audio_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption, senid in zip(self.vid2caption[vid], self.vid2senid[vid]):
                    self.all_train_pairs.append([vid, caption, senid])
            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        self.vid2senid   = defaultdict(list)

        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
            senid = annotation['sen_id']
            self.vid2senid[vid].append(senid)
