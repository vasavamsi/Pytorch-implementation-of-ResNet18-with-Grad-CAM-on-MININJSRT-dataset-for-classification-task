import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Function
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# As MINIJSRT is a light weight dataset, we are using the CPU for creating grad cam heatmaps

# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_module = self.model._modules.get(self.target_layer)
        forward_hook_handle = target_module.register_forward_hook(forward_hook)
        backward_hook_handle = target_module.register_backward_hook(backward_hook)
        self.handles = [forward_hook_handle, backward_hook_handle]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output)

        output[0, target_class].backward(retain_graph=True)
        gradients = self.gradients[0]
        activations = self.activations[0]

        alpha = gradients.mean(dim=(1, 2), keepdim=True)
        weighted_activations = (alpha * activations).sum(dim=0)

        return weighted_activations

# Load pre-trained ResNet-18
import torchvision.models as models
model = models.resnet18(pretrained = True)
num_features = model.fc.in_features     #extract fc layers features
model.fc = nn.Linear(num_features, 4)

#Replace the path accordingly
checkpoint = torch.load('./MINIJSRT_ResNet18_Gender/MINIJSRT_ResNet18_Gender_10.model', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

#Tranforming the dataset
import torchvision
from torchvision.transforms import transforms

transformer = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.Pad([49,49,49,49], padding_mode = 'symmetric'),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])

# Load and preprocess an image
image_path = './Gender01_RGB/test/up/41.png'
image = cv2.imread(image_path)
# image = Image.open(image_path)
print(image.shape)
input_tensor = transformer(image).unsqueeze(0)

# Initialize GradCAM
grad_cam = GradCAM(model, target_layer='layer4')

# Get GradCAM heatmap
heatmap = grad_cam(input_tensor)

# Normalize the heatmap
heatmap = heatmap.detach().numpy()
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

# Upsample heatmap to match the original image size
heatmap = np.uint8(255 * heatmap)
heatmap = np.uint8(Image.fromarray(heatmap).resize((128,128), Image.BILINEAR))

# Apply heatmap on the original image
heatmap = plt.cm.jet(heatmap)[..., :3]
# superimposed_img = heatmap * 0.4 + np.array(image)

# Display the original image, heatmap, and superimposed image
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(122)
plt.imshow(heatmap)
plt.title('Heatmap')
# plt.subplot(133)
# plt.imshow(superimposed_img)
# plt.title('Superimposed')
plt.show()
