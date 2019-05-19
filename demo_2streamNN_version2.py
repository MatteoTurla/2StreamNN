#!/usr/bin/env python
import os
import time
import pdb
import math

import torch
import skvideo.io

import i3d.tools as i3d_tools
import tools as tools

import docker
from pathlib import Path

from i3d.i3dpt import I3D
from stgcn.st_gcn import Model
from TwoStreamNN import TwoStreamNN

#per ora funziona quando c'è solo una persona, video deve poter stare in memoria, non sono presenti classi di rigetto
#quindi anche se non si fa nulla il sitema darà in output qualche azione
class Demo():
    def __init__(self, cluster, demo_dir_folder, video_name, output_name, modality, debug=True, openpose=True):
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
        
        label_name_path = '/home/Dataset/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        #estrazione input
        video_path = os.path.join(demo_dir_folder, video_name)
        output_path = os.path.join(demo_dir_folder, output_name)
 
        if openpose:
            openpose_folder = self._openpose()
        else:
            openpose_folder = os.path.join(self.demo_dir_folder, 'openpose')
        
        model_2stream = self._loadingModels(model_weight_stgcn, model_weight_i3d, num_classes)

        self.video = tools.utils.get_video_frames(video_path)
        video_height, video_width, video_channel = self.video[0].shape
        video_len = len(self.video)

        pose_info = tools.utils.openpose_parser(openpose_folder)
        pose_norm = tools.utils.normalize_openpose(pose_info, video_height, video_width)
        #multiple person non ancora gestito
        pose_info = pose_info[:,:,:,0].unsqueeze(3)
        pose_norm = pose_norm[:,:,:,0].unsqueeze(3)

        n_person = pose_norm.shape[3]
        ln = [[] for _ in range(n_person)]
        lp = [[] for _ in range(n_person)]
        
        for p in range(n_person):
            print('person n:', p)
            start = 0
            end = 32
            label_name_sequence = []
            label_prob_sequence = []
            pose_info_p = pose_info[:,:,:,p].unsqueeze(3)
            pose_norm_p = pose_norm[:,:,:,p].unsqueeze(3)

            while end < video_len:
                print("generating pose")
                pose_tensor = pose_norm_p[:,start:end,:,:]
                print("generating box crop")
                pose_box = pose_info_p[:,start:end]
                score = pose_box[2].mean()
                if score > 0.3:
                    
                    pose_box = pose_box[:, pose_box[2]>0.3]
                    x1 = pose_box[0].min().item() // 1 - 10
                    y1 = pose_box[1].min().item() // 1 - 10
                    x2 = pose_box[0].max().item() // 1 + 10
                    y2 = pose_box[1].max().item() // 1 + 10
                    box = (x1, y1, x2, y2)
                    print("generating i3d video")
                    video_tensor = i3d_tools.utils.transform_video_crop(self.video[start:end], box)
                    if debug:
                        i3d_tools.utils.save_transform_crop(self.video[start:end], box, os.path.join(self.demo_dir_folder, 'i3d_vis.mp4'))

                    pose_tensor = pose_tensor[:,::downsample_stgcn,:,:].unsqueeze(0)
                    video_tensor = video_tensor[:,::downsample_i3d,:,:].unsqueeze(0)
                    
                    print('\nForward...')
                    output = self._forward(model_2stream, pose_tensor, video_tensor)
                    prob, label = self.get_label(label_name, offset, output)
                    label_prob_sequence.append(prob.item())
                    label_name_sequence.append(label)
                    print('Done.')
                
                start = start + 8
                end = end + 8   
            ln[p] = label_name_sequence
            lp[p] = label_prob_sequence

        print('\nVisualization...')
        
        edge = model_2stream.stgcn.graph.edge
        images = tools.utils.visualize_output(
            pose_norm, edge, self.video, ln, lp)
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
        #estrae keypoint e li salve nella cartella openpose in demo_dir
        #return cartella contente file json per ogni frame
        openpose_dir = os.path.join(self.demo_dir_folder, 'openpose')

        if os.path.exists(openpose_dir):
            os.system('rm -f {}/*'.format(openpose_dir))
        else:
            os.system('mkdir {}'.format(openpose_dir))

        #monto cartella contente video nel container docker
        docker_volume = Path(self.demo_dir_folder).absolute()
        cmd = "./build/examples/openpose/openpose.bin --video ../data/{}  \
                 --write_keypoint_json  ../data/{} --tracking 0 --number_people_max 3 --no_display".format(self.video_name, 'openpose')
        volumes = {docker_volume: {'bind': '/data', 'mode': 'rw'}}
        client = docker.from_env()
        print("extracting skeletons keypoints")
        result = client.containers.run("gemfield/openpose:latest", cmd, runtime="nvidia", volumes=volumes)
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
        model_2stream.cuda()
        return model_2stream

    def _forward(self, model, pose, i3d_video):
        pose = pose.cuda()
        i3d_video = i3d_video.cuda()
        model.eval()

        with torch.no_grad():
            output = model.forward_mean(pose, i3d_video)
        
        pose.detach().cpu()
        i3d_video.detach().cpu()
        output.detach().cpu()
        return output

    def get_label(self, label_name, offset, output):
        prob, index = output.max(1)
        return prob, label_name[index+offset]

if __name__ == '__main__':
    cluster = 'basic'
    demo_dir_folder = '../Demo/2_person'
    video_name = 'video.avi'
    output_name = '2streamNN.mp4'
    modality = '2streamNN'
    Demo(cluster, demo_dir_folder, video_name, output_name, modality, debug=True, openpose=True)




