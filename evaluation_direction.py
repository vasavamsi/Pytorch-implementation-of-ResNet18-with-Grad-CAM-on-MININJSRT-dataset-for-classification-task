import torch
from torchmetrics.classification import MulticlassAUROC #install torchmetrics library
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metricss
import os
from torch.autograd import Variable

# selecting the device
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Transforming the dataset
import torchvision
from torchvision.transforms import transforms

transformer = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.Pad([49,49,49,49], padding_mode = 'symmetric'),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])


#Dataloader
train_path = "./Directions01_RGB/train" #Change the path to training set directory
val_path = "./Directions01_RGB/val" #Change the path to validation set directory
test_path = "./Directions01_RGB/test" #Change the path to testing set directory

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path,transformer),
    batch_size = 8, shuffle=True
)
val_loader = DataLoader(
    torchvision.datasets.ImageFolder(val_path,transformer),
    batch_size = 8, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path,transformer),
    batch_size = 8, shuffle=True
)

#importing the resnet18 pretrained model from the torchvision models

net = models.resnet18(pretrained = True)
num_features = net.fc.in_features     #extract fc layers features
net.fc = nn.Linear(num_features, 4) #change final layers to have 4 dim output for 4 classes [Up,Down,Left,Right]
net = net.to(device)

test_count = len(glob.glob('./Direction01_RGB/test/**/*.png')) #replace the path to test data
checkpoint = torch.load('./MINIJSRT_ResNet18_Direction/MINIJSRT_ResNet18_Direction_10.model')  #replace the path to the checkpoint
net.load_state_dict(checkpoint)

net.eval()

#calculating the AUC on test data
y = []
y_pred = []
for i, (images,labels) in enumerate(val_loader):
  if torch.cuda.is_available():
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())

    outputs=net(images)
    outputs=torch.sigmoid(outputs)
    # _,prediction = torch.max(outputs.data,1)

    y.extend(labels.tolist())
    y_pred.extend(outputs.tolist())

y = torch.tensor(y)
y_pred = torch.tensor(y_pred)

metric = MulticlassAUROC(num_classes=4, average=None, thresholds=None)
auc = metric(y_pred, y)
print('AUC for all the classes are as follows')
print(auc)

#Calculating Accuracy, Precision and Recall on test data

y = []
y_pred = []
for i, (images,labels) in enumerate(val_loader):
  if torch.cuda.is_available():
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())

    outputs=net(images)
    outputs=torch.sigmoid(outputs)
    _,prediction = torch.max(outputs.data,1)

    y.extend(labels.tolist())
    y_pred.extend(prediction.tolist())

y = torch.tensor(y)
y_pred = torch.tensor(y_pred)

classes = os.listdir('./Directions01_RGB/test')
print('targets :', y)
print('predictions :', y_pred)
print('accuracy :', accuracy_score(y, y_pred))
print('precision :', precision_score(y, y_pred, average='macro'))
print('recall :', recall_score(y, y_pred, average='macro'))
