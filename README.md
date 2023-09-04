# Training-ResNet18-on-MINIJSRT-dataset-alongwith-Grad-CAM-implementation
This repo includes the Python code to Train ResNet18 on MINIJSRT dataset for Direction classification and Gender classification tasks. It also includes the pre-trained weights and Grad-CAM implementation (activation Heatmap generation).

### **Required installations**

> torch, 
> scikit, 
> torchvision, 
> torchmetrics, 
> cv2, 
> matplotlib, 
> sklearn.

### **Pre-requisites**

Download Direction classification and Gender classfication dataset from MINIJSRT database. follow this link to accomplish the same: http://imgcom.jsrt.or.jp/minijsrtdb/. Split the dataset into train, validation and test sets. 

## <u>**Direction Classification**</u>

train_direction, evaluation_direction and grad_cam direction can be used to train, evaluate and generate heatmaps accordingly.

