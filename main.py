import sys
import yaml
from trainer import *
from data_utils import get_data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='',
                        help='config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    parser.add_argument('--baseline_vp', default=False, help='run baseline prompts experiment')

    args = parser.parse_args()

    return args



def main_train(config, pretrained_path, save_path, device, baseline_vp):
    encoder_config = config['encoder_config']
    decoder_config = config['decoder_config']
    prompt_encoder_config = config['prompt_encoder_config']
    blackbox_config = config['blackbox_config']
    optim_config = config['optimizer_config']
    data_config = config['data_config']
    train_config = config['train_config']
    
    dataset_dict, dataloader_dict, _ = get_data(data_config)

    #train
    train(
        # dataloader_dict=dataloader_dict,
        dataset_dict=dataset_dict,
        encoder_config=encoder_config,
        prompt_encoder_config=prompt_encoder_config,
        decoder_config=decoder_config,
        blackbox_config=blackbox_config,
        optim_config=optim_config,
        train_config=train_config,
        device=device,
        pretrained_path=pretrained_path,
        save_path=save_path,
        baseline_expts=baseline_vp
    )

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # # for training the model
    main_train(config, args.pretrained_path, args.save_path, device=args.device, baseline_vp = args.baseline_vp)
