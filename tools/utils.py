from pathlib import Path
import json
import pdb
import torch
import skvideo.io
import cv2
import numpy as np
import pdb

def openpose_parser(snippets_dir):
    p = Path(snippets_dir)
    files = sorted(p.glob('*.json'))
    depth = len(files)
    npeoples = 0
    for path in files:
        json_path = str(path)
        data = json.load(open(json_path))
        n = len(data['people'])
        if n > npeoples:
            npeoples = n
    data_openpose = torch.zeros(3,depth,18,npeoples)

    for index, path in enumerate(files):
        json_path = str(path)
        data = json.load(open(json_path))
        
        for p_index, person in enumerate(data['people']):
            keypoints = person['pose_keypoints']
            j = 0
            for i in range(0, len(keypoints), 3):
                if keypoints[i+2] > 0.3:  
                    data_openpose[0,index,j,p_index] = keypoints[i]
                    data_openpose[1,index,j,p_index] = keypoints[i+1]
                    data_openpose[2,index,j,p_index] = keypoints[i+2]
                j += 1

    return data_openpose

def normalize_openpose(data, height, width):
    normalize = data.clone()
    normalize[0] = data[0] / height
    normalize[1] = data[1] / width
    normalize[0:2] = normalize[0:2] - 0.5
    normalize[0][normalize[2] == 0] = 0
    normalize[1][normalize[2] == 0] = 0
    return normalize

def get_video_frames(video_path):
    vread = skvideo.io.vread(video_path)
    video = []
    for frame in vread:
        video.append(frame)
    #heigth, width, channel
    return video



def visualize_output(pose,
                    edge,
                    video,
                    label_sequence,
                    label_sequence_prob,
                    height=1080):

    pose = pose.numpy()
    _, T, V, M = pose.shape
    T = len(video)
    for t in range(T):
        if t >= pose.shape[1]:
            continue

        frame = video[t]

        # image resize
        H, W, c = frame.shape
        H, W, c = frame.shape
        scale_factor = 2 * height / 1080

        # draw skeleton
        skeleton = frame * 0
        text = frame * 0
        m_offset = 0
        for m in range(M):
            score = pose[2, t, :, m].mean()
            if score < 0.3:
                continue

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi + yi == 0 or xj + yj == 0:
                    continue
                else:
                    xi = int((xi + 0.5) * H)
                    yi = int((yi + 0.5) * W)
                    xj = int((xj + 0.5) * H)
                    yj = int((yj + 0.5) * W)
                cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255),
                         int(np.ceil(2 * scale_factor)))

            if t // 8 < len(label_sequence[m]):
                if label_sequence_prob[m][t // 8] > 0.8:
                    body_label = label_sequence[m][t // 8]
                    cv2.putText(text, body_label, (50,50+m_offset),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.,
                                (255, 255, 255))
            m_offset += 50

        rgb_result = frame.astype(float) * 0.5
        rgb_result += skeleton.astype(float) * 0.25
        rgb_result += text.astype(float)
        rgb_result[rgb_result > 255] = 255
        rgb_result.astype(np.uint8)

        yield rgb_result

def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5), int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
            (255,255,255))
    cv2.putText(img, text, *params)

