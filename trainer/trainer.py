import torch
from torch import optim
from parse_config import parse_config
from data_loader.data_loaders import MnistDataLoader
from model.model import EBMModel


class Trainer:
    def __init__(self, model, optimizer, data_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config

    def train(self):
        for epoch in range(1, self.config['trainer']['epochs'] + 1):
            avg_loss = self._train_epoch(epoch)
            # 각 epoch이 끝날 때 모델 저장
            if epoch % self.config['trainer']['save_period'] == 0:
                self.save_checkpoint(epoch, avg_loss)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.data_loader)
        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(torch.float32)
            self.optimizer.zero_grad()

            # Gibbs 샘플링을 이용한 대조 발산
            v_0 = data
            v_k = self.model.generate_samples(v_0, steps=1, step_size=10, noise=0.005)

            # Energy 계산
            energy_v_0 = self.model.energy(v_0)
            energy_v_k = self.model.energy(v_k)

            # 손실 계산 및 역전파
            loss = -(energy_v_0 - energy_v_k).mean()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.data_loader.dataset)} ({100. * batch_idx / len(self.data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, loss):
        checkpoint_path = f"{self.config['trainer']['save_dir']}/epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)


def main(config):
    # 데이터 로더 설정
    data_loader = MnistDataLoader(data_dir=config['data_loader']['args']['data_dir'],
                                  batch_size=config['data_loader']['args']['batch_size'],
                                  shuffle=config['data_loader']['args']['shuffle'],
                                  validation_split=config['data_loader']['args']['validation_split'],
                                  num_workers=config['data_loader']['args']['num_workers'],
                                  train=True)

    # 모델 설정
    model = EBMModel()
    optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'])

    # 트레이너 설정 및 훈련
    trainer = Trainer(model, optimizer, data_loader, config)
    trainer.train()


if __name__ == '__main__':
    config = parse_config('config.json')  # config.json 파일에서 설정 로드
    main(config)
