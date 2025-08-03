import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_stochastic import Trainer
from config.all_config import gen_log
import datetime
from thop import profile
import time


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
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
        config.model_path = os.path.join(config.output_dir, config.exp_name, config.datetime)
        msg = f'model pth = {config.model_path}'
        gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
        gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')


    # seed
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("pretrained/openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # data I/O
    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    

    model = model.to("cuda")  # 需要放到 GPU上
    model.eval()

    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            print('load best model')
            trainer.load_checkpoint("model_best.pth")

    trainer.validate()


if __name__ == '__main__':
    main()

