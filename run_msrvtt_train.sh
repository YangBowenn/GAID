#! /bin/zsh
python -m torch.distributed.launch --nproc_per_node=1 --master_port 19676 \
train.py  \
    --arch=clip_stochastic \
    --exp_name=MSR-VTT-9k \
    --videos_dir=/home/ubuntu/data16TB/Datasets/msrvtt/MSRVTT/videos/compressed \
    --audios_dir=./data/MSRVTT_audio_features \
    --batch_size=32 \
    --clip_arch=ViT-B/32 \
    --noclip_lr=3e-5 \
    --transformer_dropout=0.3 \
    --dataset_name=MSRVTT \
    --msrvtt_train_file=9k \
    --num_epochs=5 \
    --num_frames=12 \
    --support_loss_weight=0.8 \
    --audio_encoder=whisper \
    --stochasic_trials=20 \
    --evals_per_epoch=10 \
    --DSL

    # --embed_dim=768 \