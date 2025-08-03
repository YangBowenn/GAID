#! /bin/zsh
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 \
test.py  \
    --clip_arch=ViT-B/32 \
    --datetime=2025_07_23_17_29_01 \
    --arch=clip_stochastic \
    --videos_dir=/home/ubuntu/data16TB/Datasets/msrvtt/MSRVTT/videos/compressed \
    --audios_dir=./data/MSRVTT_audio_features \
    --batch_size=32 \
    --noclip_lr=3e-5 \
    --load_epoch=-1 \
    --transformer_dropout=0.3 \
    --dataset_name=MSRVTT \
    --msrvtt_train_file=9k \
    --stochasic_trials=20 \
    --support_loss_weight=0.8 \
    --exp_name=MSR-VTT-9k \
    --audio_encoder=whisper \
    --save_eval \
    --metric='t2v' \
    --DSL