import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision
import os
import time
import copy
import matplotlib.pyplot as plt
import sys
import numpy as np

from i3d.i3dpt import I3D
from stgcn.st_gcn import Model
from TwoStreamNN import TwoStreamNN
from myDataset import Dataset

def save_checkpoint(epoch, model, optimizer, scheduler, path):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, path)
save_checkpoint_bool = True

dist_dir = 'stats/basic'
os.makedirs(dist_dir)
if save_checkpoint_bool:
    checkpoint_path = dist_dir+"/checkpoint"
    os.makedirs(checkpoint_path)

best_model_path = dist_dir+"/best_model.pt"
readme_path = dist_dir+"/readme.txt"
confusion_matrix_path = dist_dir+"/confusion_matrix.txt"
plot_path = dist_dir+"/plot.png"

nframes = 16
downsample = 2
cluster = 'basic'
batch_size = 8
num_epochs = 5


dataset_train = Dataset('../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/basic/{}'.format('train'), 
                        '../Dataset/aggregorio_balanced/aggregorio_videos_pytorch/basic/{}'.format('train'),
                        n_frames=nframes, campionamento=downsample)
loader_train = loader_train = data.DataLoader(
                                dataset_train,
                                batch_size=batch_size,
                            	shuffle=True,
                            	pin_memory=True,
                                num_workers = 4
                            )

dataset_test = Dataset('../Dataset/aggregorio_skeletons_numpy/basic/{}'.format('test'), 
                        '../Dataset/aggregorio_videos_pytorch/basic/{}'.format('test'),
                        n_frames=nframes, campionamento=downsample)
loader_test = loader_train = data.DataLoader(
                                dataset_test,
                                batch_size=batch_size,
                            	shuffle=False,
                            	pin_memory=True,
                                num_workers = 4
                            )

nclasses = len(dataset_train.classes)

print('loading i3d')
dropout_prob = 0.
model_i3d = I3D(num_classes=nclasses, modality='rgb', dropout_prob=dropout_prob)
model_i3d.load_state_dict(torch.load('i3d/best_model/{}.pt'.format(cluster)))
model_i3d.cuda()

print('loading stgcn')
model_args = {
  'in_channels': 3,
  'num_class': nclasses,
  'edge_importance_weighting': True,
  'graph_args': {
    'layout': 'openpose',
    'strategy': 'spatial'
  }
}
model_stgcn = Model(model_args['in_channels'], model_args['num_class'],
                model_args['graph_args'], model_args['edge_importance_weighting'])
model_stgcn.load_state_dict(torch.load('stgcn/best_model/{}.pt'.format(cluster)))
model_stgcn.cuda()

model_2stream = TwoStreamNN(model_i3d, model_stgcn, nclasses)
model_2stream = model_2stream.cuda()


for param in model_2stream.parameters():
    param.requires_grad = False
for param in model_2stream.fc1.parameters():
    param.requires_grad = True
for param in model_2stream.fc2.parameters():
    param.requires_grad = True
model_2stream.cuda()

print("parameters to learn:")
params_to_update = []
params_to_update_name = []
for name,param in model_2stream.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_to_update_name.append(name)
        print("\t",name)

w_decay = 0.
optimizer = optim.SGD(params_to_update, lr=0.5, momentum=0.9, weight_decay=w_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,50], gamma=0.1, last_epoch=-1)

since = time.time()
train_acc_history = []
val_acc_history = []
best_model_wts = copy.deepcopy(model_2stream.state_dict())
best_acc = 0.0
best_epoch = 0
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    scheduler.step()
    model_2stream.train()
    running_loss, correct = 0.0, 0

    for pose, video, label in loader_train:
        pose = pose.cuda()
        video = video.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        y_ = model_2stream(pose, video)
              
        loss = criterion(y_, label)
        loss.backward()
        optimizer.step()

        _, y_label_ = torch.max(y_, 1)
        correct += (y_label_ == label).sum().item()
        running_loss += loss.item() * pose.shape[0]

    print(f"    train accuracy: {correct/len(loader_train.dataset):0.3f}")
    print(f"    train loss: {running_loss/len(loader_train.dataset):0.3f}")

    train_acc = correct/len(loader_train.dataset)
    train_acc_history.append(train_acc)

    #validation
    model_2stream.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for pose, video, label in loader_test:
            pose = pose.cuda()
            video = video.cuda()
            label = label.cuda()

            y_ = model_2stream(pose, video)
 
            _, y_label_ = torch.max(y_, 1)
            correct += (y_label_ == label).sum().item()

        print(f"    validation accuracy: {correct/len(loader_test.dataset):0.3f}")
        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()


        epoch_acc = correct/len(loader_test.dataset)
        val_acc_history.append(epoch_acc)

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model_2stream.state_dict())
        
        if save_checkpoint_bool and epoch % 10 == 0:
            print("save model")
            save_checkpoint(epoch, model_2stream, optimizer, scheduler, checkpoint_path+"/{}_epoch.pt".format(epoch))


model_2stream.load_state_dict(best_model_wts)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

torch.save(model_2stream.state_dict(), best_model_path)

#matrice confusione
print("Calc confusion matrix")
model_2stream.eval()
conf_matrix = torch.zeros(nclasses, nclasses)
with torch.no_grad():
    for pose, video, label in loader_test:
        pose = pose.cuda()
        video = video.cuda()
        label = label.cuda()

        y_ = model_2stream(pose, video)

        _, y_label_ = torch.max(y_, 1)

        for i in range(len(label)):
            conf_matrix[label[i].item(), y_label_[i].item()] += 1


_bins = dataset_test.bincount()
bins = [_bin.item() for _bin in _bins]
for i in range(nclasses):
    for j in range(nclasses):
        conf_matrix[i][j] = conf_matrix[i][j]/bins[i]

mean_tot = 0
for i in range(nclasses):
    mean_tot += conf_matrix[i][i].item()
mean_tot /= nclasses

#ßaving some stats

orig_stdout = sys.stdout
with open(readme_path, 'w+') as f:
    sys.stdout = f
    dataset_train.print()
    dataset_test.print()
    print(params_to_update_name)
    print("batch size:", batch_size)
    print("numero epoche:", num_epochs)
    print("best model at epoch:", best_epoch)
    print("drop out prob:", dropout_prob)
    print(optimizer)
    print(criterion)
    print(scheduler.state_dict())
sys.stdout = orig_stdout


train_hist = np.array(train_acc_history)
val_hist = np.array(val_acc_history)

fig = plt.figure(figsize=(19.2,10.8), dpi=100)
plt.title("Validation vs Train")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,num_epochs+1),train_hist,label="Train")
plt.plot(range(1,num_epochs+1),val_hist,label="Test")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
fig.savefig(plot_path)

      
orig_stdout = sys.stdout
with open(confusion_matrix_path, 'w+') as f:
    sys.stdout = f
    print(conf_matrix)
    print()
    print('accuracy:', mean_tot)
sys.stdout = orig_stdout

