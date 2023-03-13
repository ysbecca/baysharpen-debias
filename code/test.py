from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import alexnet
import torch
import numpy as np

from typing import Any

import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url


__all__ = ["AlexNet", "alexnet"]


model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def alexnet2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
    return model


from PIL import Image
from torchvision import transforms


model_count = 5
models = [alexnet2(num_classes=2) for i in range(model_count)]

print(f"[INFO] {models[0].features[-3]}")


target_layers = [models[i].features[-3] for i in range(model_count)]

data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

resize_transform = transforms.Compose([
            transforms.Resize(224),
        ])

rgb_im = Image.open("../notebooks/baby.jpg")
im = data_transform(rgb_im)

input_tensor = torch.stack([im, im, im, im])# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

print(f"[INFO] input_tensor.shape {input_tensor.shape}")


# Construct the CAM object once, and then re-use it on many images:
cams = [GradCAM(model=models[i], target_layers=[models[i].features[-3]], use_cuda=False) for i in range(model_count)]

print(cams)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cams[0](input_tensor=input_tensor, targets=None)

print(f"[INFO] grayscale_cam.shape {grayscale_cam.shape}")

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

# grayscale_cam = [grayscale_cam, grayscale_cam]
# rgb_ims = [np.array(rgb_im)/255., np.array(rgb_im)/255.]
visualization = show_cam_on_image(np.array(rgb_im)/255., grayscale_cam, use_rgb=True)


print(f"[INFO] visualization.shape {visualization.shape}")
