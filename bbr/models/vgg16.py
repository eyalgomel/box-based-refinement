import torch
import torch.nn as nn
from torchvision import models


class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.Org_model = models.vgg16(pretrained=True)
        self.full_features = [64, 128, 256, 512, 512]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        self.layers = [3, 4, 8, 15, 22, 29]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = []
        for i in range(30):
            x = self.Org_model.features[i](x)
            if i in self.layers:
                out.append(x)
        return out


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.Org_model = models.vgg16(pretrained=True).features[:23]
        self.full_features = [64, 128, 256, 512, 512]
        for param in self.Org_model.parameters():
            param.requires_grad = True
        self.layers = [3, 4, 8, 15, 22]
        for param in self.Org_model.parameters():
            param.requires_grad = True

    def forward(self, x):
        for i in range(23):
            x = self.Org_model[i](x)
        return x


class BboxRegressor(nn.Module):
    def __init__(self, num_boxes=4, num_classes=1, pretrained=True):
        super(BboxRegressor, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(
            4096, (num_classes + 5) * num_boxes
        )  # 1 for background, 4 for bbox coordinates
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], self.num_boxes, (self.num_classes + 5))  # (B, box+confidence, num_boxes)
        pred_bboxes = x[:, :, :4].sigmoid()
        pred_labels = x[:, :, 4:]

        return pred_bboxes, pred_labels


class BboxRegressorResnet50DINO(nn.Module):
    def __init__(self, num_boxes=4, num_classes=1):
        super(BboxRegressorResnet50DINO, self).__init__()
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        self.model.fc = nn.Linear(2048, (num_classes + 5) * num_boxes)  # 1 for background, 4 for bbox coordinates
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], self.num_boxes, (self.num_classes + 5))  # (B, box+confidence, num_boxes)
        pred_bboxes = x[:, :, :4].sigmoid()
        pred_labels = x[:, :, 4:]
        return pred_bboxes, pred_labels
