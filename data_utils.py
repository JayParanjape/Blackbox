from datasets.glas import GLAS_Dataset
from datasets.kvasirseg import KVASIRSEG_Dataset
from torch.utils.data import DataLoader

def get_data(data_config):
    dataset_dict = {}
    dataset_sizes = {}
    dataloader_dict = {}

    if data_config['name']=='GLAS':
        dataset_dict['train'] = GLAS_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = GLAS_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = GLAS_Dataset(data_config, shuffle_list=True, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)

    elif data_config['name']=='KVASIRSEG':
        dataset_dict['train'] = KVASIRSEG_Dataset(data_config, shuffle_list=True, is_train=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])
        dataset_dict['val'] = KVASIRSEG_Dataset(data_config, shuffle_list=False, apply_norm=data_config['use_norm'], is_train=False, no_text_mode=data_config['no_text_mode'])
        dataset_dict['test'] = KVASIRSEG_Dataset(data_config, shuffle_list=True, is_train=False, is_test=True, apply_norm=data_config['use_norm'], no_text_mode=data_config['no_text_mode'])

        dataset_sizes['train'] = len(dataset_dict['train'])
        dataset_sizes['val'] = len(dataset_dict['val'])
        dataset_sizes['test'] = len(dataset_dict['test'])

        dataloader_dict['train'] = DataLoader(dataset_dict['train'], data_config['batch_size'], shuffle=True)
        dataloader_dict['val'] = DataLoader(dataset_dict['val'], data_config['batch_size'], shuffle=True)



    return dataset_dict, dataloader_dict, dataset_sizes