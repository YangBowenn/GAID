#! /bin/zsh
python -m torch.distributed.launch --nproc_per_node=2 --master_port 19679 \
train.py  \
    --arch=clip_stochastic \
    --exp_name=VATEX \
    --videos_dir=/home/ubuntu/data16TB/Datasets/vatex/videos \
    --audios_dir=/home/ubuntu/data16TB/Datasets/vatex/audios \
    --batch_size=32 \
    --clip_arch=ViT-B/32 \
    --noclip_lr=1e-5 \
    --transformer_dropout=0.4 \
    --dataset_name=VATEX \
    --num_workers=8 \
    --audio_encoder=whisper \
    --num_epochs=5 \
    --DSL