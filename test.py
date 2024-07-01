import torch
from parse_config import parse_config
from model.model import EBMModel
from data_loader.data_loaders import MnistDataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# 임의의 IMAGE_SIZE 값 설정
IMAGE_SIZE = 32


def main(config):
    # 데이터 로더 설정
    data_loader = MnistDataLoader(data_dir=config['data_loader']['args']['data_dir'],
                                  batch_size=config['data_loader']['args']['batch_size'],
                                  shuffle=config['data_loader']['args']['shuffle'],
                                  validation_split=config['data_loader']['args']['validation_split'],
                                  num_workers=config['data_loader']['args']['num_workers'],
                                  train=False)

    # 모델 설정
    model = EBMModel()
    checkpoint = torch.load(config['trainer']['save_dir'] + '/epoch_30.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # 시각화할 이미지 수
    num_images_to_visualize = 5

    for idx, (data, target) in enumerate(data_loader):
        if idx >= num_images_to_visualize:
            break

        # 원본 이미지 시각화
        plt.figure(figsize=(15, 5))
        original_image = np.squeeze(data.numpy(), axis=1)
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(original_image[0], cmap='gray')
        plt.axis('off')

        # 랜덤한 위치에서 1/4 크기로 이미지 마스킹
        masked_image = random_mask_quarter(original_image[0])
        print(masked_image.shape)
        # 생성된 이미지 시각화
        plt.subplot(1, 3, 2)
        plt.title('Masked Image')
        plt.imshow(masked_image.numpy(), cmap='gray')
        plt.axis('off')
        # 복원된 이미지 얻기
        masked_image = masked_image.unsqueeze(0).unsqueeze(0)
        print(masked_image.shape)
        generated_image = generate_samples(model, masked_image, steps=1000, step_size=1,
                                           noise=0.005, return_img_per_step=False).detach().cpu().numpy()

        print(generated_image.shape)
        # 복원된 이미지 시각화
        plt.subplot(1, 3, 3)
        plt.title('Generated Image')
        plt.imshow(generated_image[0].squeeze(0), cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def generate_samples(model, inp_imgs, steps, step_size, noise, return_img_per_step=False):
    imgs_per_step = []
    for i in range(steps):
        inp_imgs += torch.randn_like(inp_imgs) * noise
        inp_imgs = torch.clamp(inp_imgs, -1.0, 1.0)

        inp_imgs.requires_grad_(True)  # requires_grad 설정

        out_score = model(inp_imgs)
        grads = torch.autograd.grad(outputs=out_score, inputs=inp_imgs,
                                    grad_outputs=torch.ones_like(out_score),
                                    create_graph=True, retain_graph=True)[0]

        grads = torch.clamp(grads, -0.03, 0.03)
        inp_imgs = inp_imgs + step_size * grads
        inp_imgs = torch.clamp(inp_imgs, -1.0, 1.0)

        if return_img_per_step:
            imgs_per_step.append(inp_imgs.clone())

    if return_img_per_step:
        return torch.stack(imgs_per_step, dim=0)
    else:
        return inp_imgs


def random_mask_quarter(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()  # numpy 배열을 Torch tensor로 변환하고 float 타입으로 설정

    img_height, img_width = img.shape
    new_height = img_height // 2
    new_width = img_width // 2

    # 초기 마스크 생성
    mask = torch.ones_like(img)

    # 랜덤한 위치에서 0으로 설정할 마스크 생성
    start_h = np.random.randint(0, img_height - new_height + 1)
    start_w = np.random.randint(0, img_width - new_width + 1)
    mask[start_h:start_h + new_height, start_w:start_w + new_width] = 0

    # 마스크 적용하여 이미지 마스킹
    masked_image = img * mask

    return masked_image


if __name__ == '__main__':
    config = parse_config('config.json')
    main(config)
