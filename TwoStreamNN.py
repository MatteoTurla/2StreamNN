import torch
import torch.nn.functional as F

class TwoStreamNN(torch.nn.Module):
    def __init__(self, model_i3d, model_stgcn, num_classes):
        super().__init__()
        self.I3D = model_i3d
        self.stgcn = model_stgcn

        #self.fc1 = torch.nn.Linear(1024+256, 512)
        #self.fc2 = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, X_stgcn, X_i3d):
        out_stgcn, logits_stgcn = self.stgcn(X_stgcn)
        out_i3d, logits_i3d = self.I3D(X_i3d)
        out = logits_stgcn + logits_i3d
        #entropy loss do LogSoftmax function when calc loss
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





        
