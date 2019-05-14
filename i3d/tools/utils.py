from PIL import Image, ImageOps
import torch
import skvideo
import torchvision

transforms = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])
toPILImage = torchvision.transforms.ToPILImage()
resize =torchvision.transforms.Resize(224)

def padding(image):
    w, h = image.size
    if w > h:
        delta_w = 0
        delta_h = w - h
    elif w < h:
        delta_w = h - w
        delta_h = 0
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(image, padding)

def transform_video(video, points):
    video_torch = []
    for i in range(len(video)):
        frame = video[i]
        frame = toPILImage(frame)
        frame = frame.crop(points)
        frame = padding(frame)
        frame = transforms(frame)
        video_torch.append(frame)

    video_torch = torch.stack(video_torch)
    video_torch = video_torch.permute(1,0,2,3)
    return video_torch


def save_transform(video, points, path):
    video_out = []
    for i in range(len(video)):
        frame = video[i]
        frame = frame = toPILImage(frame)
        frame = frame.crop(points)
        frame = padding(frame)
        frame = resize(frame)
        video_out.append(frame)

    writer = skvideo.io.FFmpegWriter(path,
                                        outputdict={'-b': '300000000'})
    for img in video_out:
        writer.writeFrame(img)
    writer.close()
