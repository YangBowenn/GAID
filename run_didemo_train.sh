#! /bin/zsh
python -m torch.distributed.launch --nproc_per_node=2 --master_port 19677 \
train.py  \
    --arch=clip_stochastic \
    --exp_name=DiDeMo \
    --videos_dir=/home/ubuntu/data16TB/Datasets/DiDemo \
    --batch_size=32 \
    --clip_arch=ViT-B/32 \
    --num_frames=12 \
    --noclip_lr=1e-5 \
    --transformer_dropout=0.4 \
    --dataset_name=DiDeMo \
    --audio_encoder=whisper \
    --num_epochs=8 \
    --DSL 