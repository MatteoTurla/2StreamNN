import torch
import torch.nn.functional as F

class TwoStreamNN(torch.nn.Module):
    def __init__(self, model_i3d, model_stgcn, num_classes):
        super().__init__()
        self.I3D = model_i3d
        self.stgcn = model_stgcn

        self.fc1 = torch.nn.Linear(1024+256, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(1)
        self.f_softmax = torch.nn.Softmax(0)

    def forward(self, X_stgcn, X_i3d):
        out_stgcn = self.stgcn(X_stgcn).squeeze(3).squeeze(2)
        out_i3d = self.I3D(X_i3d).squeeze(4).squeeze(3).squeeze(2)
        out = torch.cat((out_stgcn, out_i3d), dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.softmax(out)
        return out

    def forward_mean(self, X_stgcn, X_i3d): 
        #use this for calc conf matrix -> pose and video must have same T
        out_i3d, out_i3d_logits = self.I3D(X_i3d)
        out_stgcn, out_stgcn_logits = self.stgcn(X_stgcn)

        out = out_i3d_logits + out_stgcn_logits
        out = self.softmax(out)
        return out

    def extract_feature(self, X_stgcn, X_i3d):
        with torch.no_grad():
            feature_stgcn, _ = self.stgcn.extract_feature(X_stgcn)
            feature_i3d = self.I3D.extract_feature(X_i3d)       
                
        return feature_stgcn, feature_i3d





        
