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

train_direction, evaluation_direction and grad_cam direction can be used to train, evaluate and generate heatmaps accordingly. We have shown the metrics obtained on test set after training the model for 10 Epochs.

AUC for 'Up' | AUC for 'Down' | AUC for 'Left' | AUC for 'Right' 
--- | --- | --- | --- 
0.95 | 0.983 | 0.97 | 0.93 

Accuracy | Precision | Recall
--- | --- | --- 
0.975 | 0.9772 | 0.975

We have implemented Grad-CAM method to understand the decision making factors of the model. The heatmaps obtained are as shown below.

![heatmaps_for_direction_classification drawio](https://github.com/vasavamsi/Training-ResNet18-on-MINIJSRT-dataset-for-classfication-alongwith-Grad-CAM-implementation/assets/58003228/f2a7a249-16d5-4416-9fdb-734b4d7778bc)

