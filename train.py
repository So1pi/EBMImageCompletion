import torch
from parse_config import parse_config
from trainer.trainer import Trainer
from data_loader.data_loaders import MnistDataLoader
from model.model import EBMModel
from torch import optim


def main(config):
    # 데이터 로더 설정
    data_loader = MnistDataLoader(data_dir= config['data_loader']['args']['data_dir'],
                                  batch_size=config['data_loader']['args']['batch_size'],
                                  shuffle=config['data_loader']['args']['shuffle'],
                                  validation_split=config['data_loader']['args']['validation_split'],
                                  num_workers=config['data_loader']['args']['num_workers'])

    # 모델 설정
    model = EBMModel()

    # 옵티마이저 설정
    optimizer = getattr(optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])
    criterion = getattr(torch.nn, config['loss'])()
    # Trainer 설정
    trainer = Trainer(model, optimizer, data_loader, config)

    # 훈련 시작
    trainer.train()


if __name__ == '__main__':
    config = parse_config('config.json')
    main(config)
