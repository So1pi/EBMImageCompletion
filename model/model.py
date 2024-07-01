import torch
import torch.nn as nn
import torch.nn.functional as F

class EBMModel(nn.Module):
    def __init__(self):
        super(EBMModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = self.flatten(x)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x

    def energy(self, x):
        return self.forward(x)

    def generate_samples(self, inp_imgs, steps, step_size, noise, return_img_per_step=False):
        imgs_per_step = []
        for i in range(steps):
            inp_imgs += torch.randn_like(inp_imgs) * noise
            inp_imgs = torch.clamp(inp_imgs, -1.0, 1.0)

            inp_imgs.requires_grad_(True)  # requires_grad 설정

            out_score = self.energy(inp_imgs)
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
