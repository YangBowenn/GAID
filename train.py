import os
import torch
import random
import numpy as np
from modules.loss import LossFactory
from config.all_config import gen_log
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from trainer.trainer_stochastic import Trainer
from modules.metrics import t2v_metrics, v2t_metrics
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
import time
import datetime
from modules.basic_utils import mkdirp

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    
    # Distributed GPUs
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200000))
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        config.world_size = torch.distributed.get_world_size()
        config.n_gpu = config.world_size
        torch.distributed.barrier()
        rank = torch.distributed.get_rank()
        config.rank = rank
    else:
        raise Exception('NO GPU!')

    if config.rank == 0:
        # @WJM: add log
        if config.datetime is None:
            rand_wait = np.random.randint(low=1, high=20)
            time.sleep(rand_wait)
            date_time = str(datetime.datetime.now())
            date_time = config.time2file_name(date_time)
        else:
            date_time = config.datetime
        config.model_path = os.path.join(config.output_dir, config.exp_name, date_time)
        print('>>>args.model_path', config.model_path)
        mkdirp(config.model_path)
        
        msg = f'model pth = {config.model_path}'
        gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
        msg = f'\nconfig={config.__dict__}\n'
        gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
        gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')
        gen_log(model_path=config.model_path, log_name='log_tot_loss', msg='Prepare to record loss values per batch ')
        gen_log(model_path=config.model_path, log_name='log_ori_loss', msg='Prepare to record loss values per batch ')
        gen_log(model_path=config.model_path, log_name='log_sup_loss', msg='Prepare to record loss values per batch ')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("pretrained/openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)


    # data I/O
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    # metric
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    # optim
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    
    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      metrics=metrics,
                      optimizer=optimizer,
                      loss=loss,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)
    
    if config.load_epoch is not None:
        trainer.load_checkpoint("model_best.pth")

    trainer.train()


if __name__ == '__main__':
    main()
