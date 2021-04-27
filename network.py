import torch
import torch.nn as nn


class SpectralNet(nn.Module):
    def __init__(self, in_channels):
        super(SpectralNet, self).__init__()
        self.spectral_cnn_3 = nn.Conv2d(in_channels, 20, 3, 1, 1)
        self.spectral_cnn_5 = nn.Conv2d(in_channels, 20, 5, 1, 2)
        self.spectral_cnn_7 = nn.Conv2d(in_channels, 20, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, spectral_x):
        spectral_feature_3 = self.spectral_cnn_3(spectral_x)
        spectral_feature_5 = self.spectral_cnn_5(spectral_x)
        spectral_feature_7 = self.spectral_cnn_7(spectral_x)
        spectral_feature = torch.cat((spectral_feature_3, spectral_feature_5, spectral_feature_7), dim=1)
        spectral_feature = self.relu(spectral_feature)
        return spectral_feature


class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        self.spatial_cnn_3 = nn.Conv2d(1, 20, 3, 1, 1)
        self.spatial_cnn_5 = nn.Conv2d(1, 20, 5, 1, 2)
        self.spatial_cnn_7 = nn.Conv2d(1, 20, 7, 1, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spatial_x):
        spatial_feature_3 = self.spatial_cnn_3(spatial_x)
        spatial_feature_5 = self.spatial_cnn_5(spatial_x)
        spatial_feature_7 = self.spatial_cnn_7(spatial_x)
        spatial_feature = torch.cat((spatial_feature_3, spatial_feature_5, spatial_feature_7), dim=1)
        spatial_feature = self.relu(spatial_feature)
        return spatial_feature


class HSID(nn.Module):
    def __init__(self, in_channels):
        super(HSID, self).__init__()
        self.spectral_net = SpectralNet(in_channels)
        self.spatial_net = SpatialNet()

        self.layer1 = nn.Sequential(nn.Conv2d(120, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(nn.Conv2d(60, 60, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv_layer_3 = nn.Conv2d(60, 15, 3, 1, 1)
        self.conv_layer_5 = nn.Conv2d(60, 15, 3, 1, 1)
        self.conv_layer_7 = nn.Conv2d(60, 15, 3, 1, 1)
        self.conv_layer_9 = nn.Conv2d(60, 15, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(60, 1, 3, 1, 1)

    def forward(self, spatial_x, spectral_x):
        spectral_feature = self.spectral_net(spectral_x)
        spatial_feature = self.spatial_net(spatial_x)
        feat = torch.cat((spectral_feature, spatial_feature), dim=1)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat_3 = self.layer3(feat)
        feat = self.layer4(feat_3)
        feat_5 = self.layer5(feat)
        feat = self.layer6(feat_5)
        feat_7 = self.layer7(feat)
        feat = self.layer8(feat_7)
        feat_9 = self.layer9(feat)

        feat_3 = self.conv_layer_3(feat_3)
        feat_5 = self.conv_layer_3(feat_5)
        feat_7 = self.conv_layer_3(feat_7)
        feat_9 = self.conv_layer_3(feat_9)

        feat = torch.cat((feat_3, feat_5, feat_7, feat_9), dim=1)
        feat = self.relu(feat)
        feat = self.conv_out(feat)
        return feat


if __name__ == '__main__':
    # device = torch.device('cuda:1')
    device = torch.device('cpu')

    x1 = torch.randn((1, 1, 200, 200)).to(device)
    x2 = torch.randn((1, 24, 200, 200)).to(device)

    model = HSID(24).to(device)

    pred = model(x1, x2)
    print(pred.shape)





