import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from torchvision.ops import box_iou, nms, MultiScaleRoIAlign
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor()] #useful for bounding box
)

image = Image.open('*.png')
image_tensor = transforms(image)
image_reshaped = image_tensor.unsqueeze(0)

image_tensor = transform(image)

box = [10, 10, 200, 200] #X_min/max, y_min/max
bbox_tensor = torch.tensor(box)
bbox_tensor = bbox_tensor.unsqueeze(0)

bbox_image = draw_bounding_boxes(
    image_tensor, bbox_tensor, width=3, colors='red'
)

transform_bbx = transforms.Compose([
    transforms.ToPILImage() 
])
pil_image = transform_bbx(bbox_image)

plt.imshow(pil_image)

#evaluation
#IoU

bbox1 =[50, 50, 100, 150]
bbox1 = torch.tensor(bbox1).unsqueeze(0)
bbox2 = [100, 100, 200, 200]
bbox2 = torch.tensor(bbox2).unsqueeze(0)

iou = box_iou(bbox1, bbox2)
print(iou)

#model predictions
test_image = Image.open('*.png')
model = resnet18()
with torch.no_grad():
    output = model(test_image)

boxes = output[0]["boxes"]
scores = output[0]["scores"]
print(boxes, scores)

#non-max seppression (nms)
#finding most accurate box
iou_threshold = 0.5
box_indices = nms(
    boxes=boxes,
    scores=scores,
    iou_threshold=iou_threshold
)
filtered_boxes = boxes[box_indices]

print("Filtered Boxes:", filtered_boxes)

vgg = vgg16(weights=VGG16_Weights.DEFAULT)
backbone = nn.Sequential(
    *list(vgg.features.children())
)

#classifier layer
class ObjectDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__(ObjectDetectorCNN, self)
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg.features.children()))
        input_features = nn.Sequential(*list(vgg.classifier.children()))[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.box_regressor = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        features = self.backbone(x)
        bboxes = self.regressor(features)
        classes = self.classifier(features)
        return bboxes, classes
    
#RPN
num_classes = 2
anchor_generator = AnchorGenerator(
    size = ((32,64,128),),
    aspect_ratios = ((0.5, 1.0, 2.0),),
)
rio_pooler = MultiScaleRoIAlign(
    featmap_names=["0"],
    output_size=7,
    sampling_ratio=2
)
backbone = torchvision.model.mobilenet_v2(weights="DEFAULT").features
backbone.out_channels = 1280

model = FasterRCNN(
    backbone=backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=rio_pooler
)

#oretrained
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

######Loss Functions
#RPN classification
rpn_cls_criterion = nn.BCEWithLogitsLoss()
#RPN regression
rpn_reg_criterion = nn.MSELoss()
#R-CNN classification
rcnn_cls_criterion = nn.CrossEntropyLoss()
#R-CNN regression 
rcnn_reg_criterion = nn.MSELoss()