#! /bin/zsh
python -m torch.distributed.launch --nproc_per_node=2 --master_port 19682 \
train.py  \
    --arch=clip_stochastic \
    --exp_name=LSMDC \
    --videos_dir=/home/ubuntu/data16TB/Datasets/LSMDC/videos_compressed \
    --audios_dir=/home/ubuntu/data16TB/Datasets/LSMDC/all_videos_audios_features \
    --batch_size=32 \
    --clip_arch=ViT-B/32 \
    --noclip_lr=1e-5 \
    --transformer_dropout=0.3 \
    --dataset_name=LSMDC \
    --stochasic_trials=20 \
    --stochastic_prior_std=3e-3 \
    --num_epochs=5 \
    --audio_encoder=whisper \
    --support_loss_weight=0.5 \
    --DSL