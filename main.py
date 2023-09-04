import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as f
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import glob

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

#Optimizer and Loss Function

optimizer = Adam(net.parameters(),lr=0.01,weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

#no. of epochs
num_epochs = 10

#calculating the size of training and testing data
train_count = len(glob.glob(train_path+'/**/*.png'))
val_count = len(glob.glob(test_path+'/**/*.png'))
test_count = len(glob.glob(test_path+'/**/*.png'))

#Model training

# Uncomment the below lines to load the checkpoint for training
# checkpoint = torch.load('./best_ResNet18_MINIJRST_Direction_49.model')
# net.load_state_dict(checkpoint)

best_accuracy=0.0

tr_loss = []
tr_acc = []
val_acc = []

for epoch in range(num_epochs):

  #Evaluation and training on training set
  net.train()
  train_accuracy=0.0
  train_loss=0.0
  for i, (images,labels) in enumerate(train_loader):
    if torch.cuda.is_available():
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())

    optimizer.zero_grad()

    outputs=net(images)
    outputs=torch.sigmoid(outputs)
    loss=loss_function(outputs,labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.cpu().data*images.size(0)
    _,prediction = torch.max(outputs.data,1)

    train_accuracy += torch.sum(prediction==labels.data)
  train_accuracy = train_accuracy/train_count
  train_loss = train_loss/train_count

  #Evaluation on validation data
  net.eval()

  val_accuracy=0.0
  y = []
  y_pred = []
  for i, (images,labels) in enumerate(val_loader):
    if torch.cuda.is_available():
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())
      # print(labels.tolist())

      outputs=net(images)
      outputs=torch.sigmoid(outputs)
      # _,prediction = outputs.data
      _,prediction = torch.max(outputs.data,1)
      # print(prediction)

      val_accuracy += torch.sum(prediction==labels.data)
    val_accuracy = val_accuracy/val_count

  # if ((epoch+1) % 5) == 0:
  model_name = './MINIJSRT_ResNet18_Direction/MINIJSRT_ResNet18_Direction_' + str(epoch + 1) + '.model'
  torch.save(net.state_dict(),model_name)
  print('model saved as : ' + model_name)

    # #Saving best model
    # if val_accuracy > best_accuracy:
    #   torch.save(net.state_dict(),'best_ResNet50_MINIJRST.model')
    #   best_accuracy = val_accuracy

  tr_loss.append(train_loss)
  tr_acc.append(train_accuracy)
  val_acc.append(val_accuracy)

  print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Val Accuracy: '+str(val_accuracy)+'')

# plotting for training loss
x = np.array(range(1,num_epochs+1))
y = np.array(tr_loss)

plt.xlabel('epoch', fontsize=10)
plt.ylabel('training loss', fontsize=10)
plt.plot(x,y)
plt.show()