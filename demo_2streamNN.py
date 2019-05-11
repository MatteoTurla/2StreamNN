#!/usr/bin/env python
import os
import time

import torch
import skvideo.io

import stgcn.tools.utils as stgcn_tools

import i3d.tools as i3d_tools

import docker
from pathlib import Path

from i3d.i3dpt import I3D
from stgcn.st_gcn import Model
from TwoStreamNN import TwoStreamNN

since = time.time()

cluster = 'alerting'
demo_dir_folder = 'demo/test_falling' #folder that contain video input
video_name = 'video.avi'
video_path = os.path.join(demo_dir_folder, video_name)
output_result_path = '{}/stgcn.mp4'.format(demo_dir_folder)

downsample_i3d = 2
downsample_stgcn = 1
model_weight_stgcn = 'stgcn/best_model/{}.pt'.format(cluster)
model_weight_i3d = 'i3d/best_model/{}.pt'.format(cluster)

#EXTRACT OPENPOSE KEYPOINTS
openpose_dir = os.path.join(demo_dir_folder, 'openpose')

if os.path.exists(openpose_dir):
    os.system('rm -f {}/*'.format(openpose_dir))
else:
    os.system('mkdir {}'.format(openpose_dir))
#monto cartella contente video nel container docker
docker_volume = Path(demo_dir_folder).absolute()
cmd = "./build/examples/openpose/openpose.bin --video ../data/{} --write_video \
        ../data/openpose.avi --write_keypoint_json  ../data/{} --no_display".format(video_name, 'openpose')
volumes = {docker_volume: {'bind': '/data', 'mode': 'rw'}}
client = docker.from_env()
print("extracting skeletons keypoints")
result = client.containers.run("mjsobrep/openpose:latest", cmd, runtime="nvidia", volumes=volumes)
print(result.decode('UTF-8'))

#EXTRACT FRAME I3D
print("extracting frames")
video = stgcn_tools.video.get_video_frames(video_path)
video_height, video_width, _ = video[0].shape
video_tensor = i3d_tools.utils.transform_video(video)
video_tensor = video_tensor[:,::downsample_i3d,:,:].unsqueeze(0).cuda()


#generate torch skeleton
print("generating json skeleton")
video_info = stgcn_tools.openpose.json_pack(
    openpose_dir, frame_width=video_width, frame_height=video_height)
pose, _ = stgcn_tools.video.video_info_parsing(video_info, num_person_out=1)
pose_tensor = torch.from_numpy(pose).float()
pose_tensor = pose_tensor[:,::downsample_stgcn,:,:].unsqueeze(0).cuda()
print("video:", len(video), "frames:", video_tensor.shape, "pose:", pose_tensor.shape)

# NON MODIFICARE  DA QUA IN POI 
label_name_path = '../Dataset/label_name.txt'
with open(label_name_path) as f:
    label_name = f.readlines()
    label_name = [line.rstrip() for line in label_name]

if cluster == 'basic':
    num_classes = 4
    offset = 0
elif cluster == 'alerting':
    num_classes = 8
    offset = 4
elif cluster == 'daily_life':
    num_classes = 7
    offset = 4+8

print('loading i3d')
model_i3d = I3D(num_classes=num_classes, modality='rgb')
model_i3d.load_state_dict(torch.load(model_weight_i3d))
model_i3d.cuda()

print('loading stgcn')
model_args = {
  'in_channels': 3,
  'num_class': num_classes,
  'edge_importance_weighting': True,
  'graph_args': {
    'layout': 'openpose',
    'strategy': 'spatial'
  }
}
model_stgcn = Model(model_args['in_channels'], model_args['num_class'],
                model_args['graph_args'], model_args['edge_importance_weighting'])
model_stgcn.load_state_dict(torch.load(model_weight_stgcn))
model_stgcn.cuda()

model_2stream = TwoStreamNN(model_i3d, model_stgcn, num_classes)
model_2stream = model_2stream.cuda()
model_2stream.eval()

output, feature = model_2stream.extract_feature(pose_tensor, video_tensor)
print("output:", output.shape)
feature = feature[0]
intensity = (feature*feature).sum(dim=0)**0.5
intensity = intensity.cpu().detach().numpy()

# visualization
print('\nVisualization...')
label_sequence = output.argmax(dim=0)
label_sequence = label_sequence.unsqueeze(1)
print(label_sequence.shape)
label_name_sequence = [[label_name[p+offset] for p in l ]for l in label_sequence]
edge = model_2stream.stgcn.graph.edge
images = stgcn_tools.visualization.stgcn_visualize_output(
    pose, edge, intensity, video, label_name_sequence)
print('Done.')

# save video
print('\nSaving...')
writer = skvideo.io.FFmpegWriter(output_result_path,
                                outputdict={'-b': '300000000'})
for img in images:
    writer.writeFrame(img)
writer.close()
print('The Demo result has been saved in {}.'.format(output_result_path))

time_elapsed = time.time() - since
print('Demo complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

