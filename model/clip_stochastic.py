import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config
from modules.transformer import Transformer
from modules.stochastic_module import StochasticText
from transformers import CLIPModel

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("pretrained/openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("pretrained/openai/clip-vit-base-patch16")
        elif config.clip_arch == 'ViT-L/14':
            self.clip = CLIPModel.from_pretrained("pretrained/openai/clip-vit-large-patch14")
        else:
            raise ValueError


        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)
        # self.pool_audios = Transformer(config)
        self.stochastic = StochasticText(config)
        # self.fusion_weight = nn.Linear(2*512, 1)
        self.reshape_audio = False
        if config.clip_arch == 'ViT-L/14':
            self.reshape_audio = True
            self.linear = nn.Linear(512, 768)
        if self.config.audio_encoder in ["wav2vec2", "ast"]:
            self.linear = nn.Linear(768, 512)
        
        # frame-level
        self.gate_fc = nn.Linear(config.embed_dim * 3, 1)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, data, return_all_frames=False, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        audio_data = data['audio']
        # print(text_data['input_ids'].shape, text_data['attention_mask'].shape, video_data.shape, audio_data.shape)
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.audio_encoder == "whisper":
            if self.reshape_audio:
                audio_data = self.linear(audio_data)
                
            audio_data = audio_data.reshape(-1, audio_data.shape[2], audio_data.shape[3])
            audio_features = audio_data.permute(0, 2, 1) # [bs, 1500, 512]
            audio_features = F.adaptive_avg_pool1d(audio_features, self.config.num_frames)
            audio_features = audio_features.permute(0, 2, 1) # [bs, #F, 512]
        
        if self.config.audio_encoder == "wav2vec2":
            audio_data = self.linear(audio_data)
            audio_data = audio_data.reshape(-1, audio_data.shape[2], audio_data.shape[3])
            audio_features = audio_data.permute(0, 2, 1) # [bs, 1500, 768]
            audio_features = F.adaptive_avg_pool1d(audio_features, self.config.num_frames)
            audio_features = audio_features.permute(0, 2, 1) # [bs, #F, 512]

        if self.config.audio_encoder == "ast":
            audio_data = self.linear(audio_data)
            audio_features = audio_data.repeat(1, self.config.num_frames, 1)
        

        text_features = self.clip.get_text_features(**text_data)
        video_features = self.clip.get_image_features(video_data)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1) # [bs, #F, 512]
        

        
        # Fame-Level
        text_expanded = text_features.unsqueeze(1).expand(-1, self.config.num_frames, -1)  # [B, F, D]
        concat_feat = torch.cat([video_features, audio_features, text_expanded], dim=-1)  # [B, F, 2D]
        gate = self.sigmoid(self.gate_fc(concat_feat))  # [B, F, 1]
        video_features = gate * audio_features + (1 - gate) * video_features
        
        
        if is_train:
    

            video_features_pooled = self.pool_frames(text_features, video_features) # [B, B, 512]

            text_features_stochstic, text_mean, log_var, all_features = \
                self.stochastic(text_features, video_features, return_feat=False)


            if return_all_frames:
                return text_features, video_features, video_features_pooled, text_features_stochstic, text_mean, log_var

            return text_features, video_features_pooled, text_features_stochstic, text_mean, log_var, all_features

        else:
              
            
            video_features_pooled = self.pool_frames(text_features, video_features)
            text_features_stochstic, _, _, all_features = \
                self.stochastic(text_features, video_features, training=False, return_feat=True)


            if return_all_frames:
                return text_features, video_features, text_features_stochstic, gate, all_features

            return text_features, video_features, text_features_stochstic
