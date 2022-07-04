import torch
import torch.nn as nn


class X3D(nn.Module):
    def __init__(self, n_classes=15, num_clips=64, model_name='x2d_m'):
        '''
        https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md
        '''
        super(X3D, self).__init__()
        assert model_name in ['x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l']
        model = torch.hub.load("facebookresearch/pytorchvideo",
                               model=model_name, pretrained=True)
        model.blocks[-1] = torch.nn.Identity()
        self.extract_fea = model
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.predictor = nn.Sequential(nn.Linear(192, 192//2), nn.BatchNorm1d(num_clips),
                                       nn.ReLU(True), nn.Linear(192//2, n_classes))

    def forward(self, x, y=None):
        x = self.extract_fea(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.predictor(x)
        return x

        # predictions = {}
        # predictions['final_output'] = x

        # return predictions
