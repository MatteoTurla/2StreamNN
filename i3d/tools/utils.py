from PIL import Image
import torch
import torchvision

transforms = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.CenterCrop(256),
                                torchvision.transforms.Resize(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])

def transform_video(video):
    video_torch = []
    for i in range(len(video)):
        frame = video[i]
        frame = transforms(frame)
        video_torch.append(frame)

    video_torch = torch.stack(video_torch)
    video_torch = video_torch.permute(1,0,2,3)
    return video_torch

    
