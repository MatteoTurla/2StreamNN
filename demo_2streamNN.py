#!/usr/bin/env python
import os
import time
import pdb
import math

import torch
import skvideo.io

import stgcn.tools.utils as stgcn_tools
import i3d.tools as i3d_tools

import docker
from pathlib import Path

from i3d.i3dpt import I3D
from stgcn.st_gcn import Model
from TwoStreamNN import TwoStreamNN

class Demo():
    def __init__(self, cluster, demo_dir_folder, video_name, output_name, modality, debug=True):
        since = time.time()
        self.cluster = cluster
        self.demo_dir_folder = demo_dir_folder
        self.video_name = video_name
        self.output_name = output_name
        self.modality = modality

        if cluster == 'basic':
            num_classes = 4
            offset = 0
        elif cluster == 'alerting':
            num_classes = 8
            offset = 4
        elif cluster == 'daily_life':
            num_classes = 7
            offset = 4+8
        else:
            raise Exception('cluster errato: {alerting, basic or daily_life}')
        
        if self.modality != 'i3d' and self.modality != 'stgcn' and self.modality != '2streamNN':
            raise Exception('modality errata: {i3d, stgcn or 2streamNN}')

        if not os.path.exists(demo_dir_folder):
            raise Exception('Demo folder non presente')  

        #COSTANTI DELLA DEMO BASATE SUL MODELLO UTILIZZATO
        downsample_i3d = 2
        downsample_stgcn = 1
        model_weight_stgcn = 'stgcn/best_model/{}.pt'.format(cluster)
        model_weight_i3d = 'i3d/best_model/{}.pt'.format(cluster)
        label_name_path = '../Dataset/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        #estrazione input
        video_path = os.path.join(demo_dir_folder, video_name)
        output_path = os.path.join(demo_dir_folder, output_name)
 
        openpose_folder = self._openpose() #os.path.join(self.demo_dir_folder, 'openpose') #

        self.video = stgcn_tools.video.get_video_frames(video_path)
        video_height, video_width, video_channel = self.video[0].shape
        
        print("generating pose")
        video_info = stgcn_tools.openpose.json_pack(openpose_folder, frame_width=video_width, frame_height=video_height)
        pose, _ = stgcn_tools.video.video_info_parsing(video_info, num_person_out=1)
        pose_tensor = torch.from_numpy(pose).float()
        x_1,y_1, x_2, y_2 = 1, 1, 0, 0
        pose_x = pose_tensor.view(3, -1)[0]
        pose_y = pose_tensor.view(3, -1)[1]
        for x in pose_x:
            if x != 0 and x < x_1:
                x_1 = x
            elif x != 0 and x > x_2:
                x_2 = x
        for y in pose_y:
            if y != 0 and y < y_1:
                y_1 = y
            elif y != 0 and y > y_2:
                y_2 = y   
        
        pose_tensor = pose_tensor[:,::downsample_stgcn,:,:].unsqueeze(0)

        x_1 = math.floor((x_1 + 0.5) * video_width)
        y_1 = math.floor((y_1 + 0.5) * video_height)
        x_2 = math.ceil((x_2 + 0.5) * video_width)
        y_2 = math.ceil((y_2 + 0.5) * video_height)
        points = (x_1,y_1,x_2,y_2)

        print("\ngenerating video")
        video_tensor = i3d_tools.utils.transform_video(self.video, points)
        if debug:
            i3d_tools.utils.save_transform(self.video, points, os.path.join(self.demo_dir_folder, 'i3d_vis.mp4'))
        video_tensor = video_tensor[:,::downsample_i3d,:,:].unsqueeze(0)
        
        print("Video:", len(self.video))
        print("Pose:", pose_tensor.shape)
        print("I3D video:", video_tensor.shape)

        model_2stream = self._loadingModels(model_weight_stgcn, model_weight_i3d, num_classes)
        print('\nExtracting Features...')
        label_prob_s, label_name_s = self._extractFeature(model_2stream, pose_tensor, video_tensor)

        labels_name = [[label_name[p+offset] for p in l ]for l in label_name_s]
        labels_prob = [[p for p in l ]for l in label_prob_s]
        print('Done.')

        print('\nVisualization...')
        edge = model_2stream.stgcn.graph.edge
        images = stgcn_tools.visualization.stgcn_visualize_output(
            pose, edge, self.video, labels_name, labels_prob)
        print('Done.')

        print('\nSaving...')
        writer = skvideo.io.FFmpegWriter(output_path,
                                        outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_path))

        time_elapsed = time.time() - since
        print('Demo complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    def _openpose(self):
        openpose_dir = os.path.join(self.demo_dir_folder, 'openpose')

        if os.path.exists(openpose_dir):
            os.system('rm -f {}/*'.format(openpose_dir))
        else:
            os.system('mkdir {}'.format(openpose_dir))
        #monto cartella contente video nel container docker
        docker_volume = Path(self.demo_dir_folder).absolute()
        cmd = "./build/examples/openpose/openpose.bin --video ../data/{}  \
                 --write_keypoint_json  ../data/{} --no_display".format(self.video_name, 'openpose')
        volumes = {docker_volume: {'bind': '/data', 'mode': 'rw'}}
        client = docker.from_env()
        print("extracting skeletons keypoints")
        result = client.containers.run("mjsobrep/openpose:latest", cmd, runtime="nvidia", volumes=volumes)
        result = result.decode('UTF-8')
        print(result)
        return openpose_dir

    def _loadingModels(self, stgcn_w, i3d_w, num_classes):
        print('loading i3d')
        model_i3d = I3D(num_classes=num_classes, modality='rgb')
        model_i3d.load_state_dict(torch.load(i3d_w))

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
        model_stgcn.load_state_dict(torch.load(stgcn_w))

        model_2stream = TwoStreamNN(model_i3d, model_stgcn, num_classes)
        return model_2stream

    def _extractFeature(self, model, pose, i3d_video):
        pose = pose.cuda()
        i3d_video = i3d_video.cuda()
        model.cuda()
        model.eval()

        out_stgcn, out_i3d = model.extract_feature(pose, i3d_video)

        f_stgcn = out_stgcn[0].sum(2).sum(2).cpu()
        f_i3d = out_i3d[0].sum(2).sum(2).cpu()
        num_classes = f_i3d.shape[0]

        t_stgcn = f_stgcn.shape[1]
        t_i3d = f_i3d.shape[1]
        video_len = len(self.video)

        t_out = (t_i3d + 1) * 4
        f_i3d = torch.cat((f_i3d, f_i3d[:,-1:]), 1)
 
        j = 0
        new_fi3d = torch.zeros((num_classes, t_out))
        for i in range(t_i3d):
            new_fi3d[:,j] = f_i3d[:,i]
            new_fi3d[:,j+1] = f_i3d[:,i]
            new_fi3d[:,j+2] = f_i3d[:,i]
            new_fi3d[:,j+3] = f_i3d[:,i]
            j += 4
        new_fi3d = new_fi3d[:,:t_stgcn]
        
        if self.modality == '2streamNN':
            output = f_stgcn[:,:video_len // 4] + new_fi3d[:,:video_len // 4]
        elif self.modality == 'i3d':
            output = new_fi3d[:,:video_len // 4]
        elif self.modality == 'stgcn':
            output = f_stgcn[:,:video_len // 4]

        label_prob, label_sequence = output.max(dim=0)
        label_prob_s, label_name_s = label_prob.unsqueeze(1), label_sequence.unsqueeze(1)

        return label_prob_s, label_name_s

if __name__ == '__main__':
    cluster = 'daily_life'
    demo_dir_folder = '../Demo/tibe'
    video_name = 'tibe_cut.avi'
    output_name = '2streamNN.mp4'
    modality = 'i3d'
    Demo(cluster, demo_dir_folder, video_name, output_name, modality)


