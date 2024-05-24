import torch 
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # фризим слои, обучать их не будем (хотя технически можно)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # задай классификационный блок
        self.clf = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.Tanh(),
            nn.Linear(256, 3)
        )

        # задай регрессионный блок
        self.box = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        ) 

    def forward(self, img):
        # задай прямой проход
        embedding = self.feature_extractor(img)
        logits = self.clf(torch.flatten(embedding, 1))
        box_coords = self.box(torch.flatten(embedding, 1))
        return logits, box_coords

       # resnet_out = self.feature_extractor(img)
       # resnet_out = resnet_out.view(resnet_out.size(0), -1)
       # pred_classes = self.clf(resnet_out)
       # pred_boxes = self.box(resnet_out)
       # print(pred_classes.shape, pred_boxes.shape)
       # return pred_classes, pred_boxes
    

    
