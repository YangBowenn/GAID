from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.didemo_dataset import DiDeMoDataset
from datasets.vatex_dataset import VATEXDataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                train_sampler = DistributedSampler(dataset)
                return DataLoader(dataset, batch_size=config.batch_size // config.n_gpu,
                           shuffle=False, num_workers=config.num_workers, sampler=train_sampler)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)
            
        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                train_sampler = DistributedSampler(dataset)
                return DataLoader(dataset, batch_size=config.batch_size // config.n_gpu, 
                                  shuffle=False, num_workers=config.num_workers, sampler=train_sampler)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "DiDeMo":
            if split_type == 'train':
                dataset = DiDeMoDataset(config, split_type, train_img_tfms)
                train_sampler = DistributedSampler(dataset)
                return DataLoader(dataset, batch_size=config.batch_size // config.n_gpu, 
                                  shuffle=False, num_workers=config.num_workers, sampler=train_sampler)
            else:
                dataset = DiDeMoDataset(config, split_type, test_img_tfms)
                shuffle = False
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)
        
        elif config.dataset_name == "VATEX":
            if split_type == 'train':
                dataset = VATEXDataset(config, split_type, train_img_tfms)
                train_sampler = DistributedSampler(dataset)
                return DataLoader(dataset, batch_size=config.batch_size // config.n_gpu, 
                                  shuffle=False, num_workers=config.num_workers, sampler=train_sampler)
            else:
                dataset = VATEXDataset(config, split_type, test_img_tfms)
                shuffle = False
                return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)
        else:
            raise NotImplementedError
