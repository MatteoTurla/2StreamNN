import torch
import torch.nn.functional as F

class TwoStreamNN(torch.nn.Module):
    def __init__(self, model_i3d, model_stgcn, num_classes):
        super().__init__()
        self.I3D = model_i3d.cuda()
        self.stgcn = model_stgcn.cuda()

        self.fc1 = torch.nn.Linear(1024+256, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(1)
        self.f_softmax = torch.nn.Softmax(0)
    def forward(self, X_stgcn, X_i3d): #, isTrain = True):
        """ chiedere a marco come funziona
        if not isTrain:
            self.I3D.eval()
            self.stgcn.eval()
        else:
            self.i3d.train()
            self.stgcn.train()
        """
        out_stgcn = self.stgcn(X_stgcn).squeeze(3).squeeze(2)
        out_i3d = self.I3D(X_i3d).squeeze(4).squeeze(3).squeeze(2)
        out = torch.cat((out_stgcn, out_i3d), dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.softmax(out)
        return out

    def forward_mean(self, X_stgcn, X_i3d): 
        self.I3D.eval()
        self.stgcn.eval()
        out_i3d, out_i3d_logits = self.I3D(X_i3d)
        out_stgcn, out_stgcn_logits = self.stgcn(X_stgcn)
        out = out_i3d_logits + out_stgcn_logits
        out = self.softmax(out)
        return out

    def extract_feature(self, X_stgcn, X_i3d):
        #stgcn work every 4 frames while i3d works every 8 frames
        self.I3D.eval()
        self.stgcn.eval()

        feature_stgcn, intensity = self.stgcn.extract_feature(X_stgcn)
        feature_i3d = self.I3D.extract_feature(X_i3d)
        print("feature stgcn:", feature_stgcn.shape, "feature i3d:", feature_i3d.shape)
        f_stgcn = feature_stgcn[0].sum(2).sum(2)
        new_fs = torch.zeros((f_stgcn.shape[0], f_stgcn.shape[1]//2))
        j = 0
        for i in range(0, f_stgcn.shape[1], 2):
            if j < new_fs.shape[1]:
                new_fs[:,j] = f_stgcn[:,i:i+2].mean(1)
                j += 1
        f_i3d = feature_i3d[0].sum(2).sum(2)
        new_i3d = torch.zeros(f_i3d.shape[0], new_fs.shape[1])
        j = 0
        for i in range(f_i3d.shape[1]):
            new_i3d[:,j] = f_i3d[:,i]
            new_i3d[:,j+1] = f_i3d[:,i]
            j += 2

        feature = new_fs + new_i3d
    
        feature = self.f_softmax(feature)

        return feature, intensity





        
