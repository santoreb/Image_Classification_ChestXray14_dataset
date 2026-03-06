import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.transforms import ToTensor,Resize,Compose,Lambda
from PIL import Image
from torch.nn.functional import interpolate
from torchvision import datasets, models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
torch.cuda.empty_cache()
from torcheval.metrics import BinaryAUROC
import timm

with open("/scratch/rpiresdo/chestxray14_convnext/dataset/train_val_list.txt", 'r') as file:
  train_list = file.readlines()

train_val_list = [x[:-1] for x in train_list]
val_list_perc = int(len(train_val_list)*0.1)
val_list = train_val_list[:val_list_perc]
train_list = train_val_list[val_list_perc:]

with open("/scratch/rpiresdo/chestxray14_convnext/dataset/test_list.txt", 'r') as file:
  test_list = file.readlines()

test_list = [x[:-1] for x in test_list]

print("Dataset size")
print(f"Training + Validation: {len(train_val_list)}")
print(f"Training: {len(train_list)}")
print(f"Validation: {len(val_list)}")
print(f"Test: {len(test_list)}")

datainfo = pd.read_csv("/scratch/rpiresdo/chestxray14_convnext/dataset/Data_Entry_2017_v2020.csv")
datainfo["Finding Labels"] = datainfo["Finding Labels"].str.lower()
datainfo["Finding Labels"] = datainfo["Finding Labels"].str.split("|")


labels = datainfo["Finding Labels"].tolist()
# print(labels)

mlb = MultiLabelBinarizer()

binarized_labels = mlb.fit_transform(labels)
feature_names = mlb.classes_
print(feature_names)

# binarized_labels_tensor = torch.tensor(binarized_labels)
# print(binarized_labels_tensor)

normal_labels = mlb.inverse_transform(binarized_labels)
# print(normal_labels)

partition = dict(train = train_list, val = val_list, test = test_list)
labels = dict(zip(datainfo["Image Index"],binarized_labels))

# print(partition)
# print(labels)

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if ID[-1] != 'g':
            ID = ID + 'g'
        image = Image.open("/scratch/rpiresdo/chestxray14_convnext/dataset/images/" + ID)
        transform = Compose([ToTensor(),Resize(size=(450,450)),Lambda(lambda x: x[:1, :, :])])
        # transform = ToTensor()
        # out = transform(image)
        # X = interpolate(out, size=128)
        X = transform(image)
        y = self.labels[ID]

        return X, y
    
import torch
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 6
          }

# Generators
train_set = Dataset(partition['train'], labels)
train_generator = torch.utils.data.DataLoader(train_set, **params)

test_set = Dataset(partition['test'], labels)
test_generator = torch.utils.data.DataLoader(test_set, **params)

val_set = Dataset(partition['val'], labels)
val_generator = torch.utils.data.DataLoader(val_set, **params)

loaders = {
    'train': train_generator,
    'val' : val_generator,
    'test': test_generator
}

#model = models.convnext_tiny(weights='DEFAULT')
#model.features[0][0] = nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
#model.classifier[2] = nn.Linear(in_features=1536, out_features=15, bias=True)
#model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
#model.classifier[2] = nn.Linear(in_features=768, out_features=15, bias=True)
#model = nn.Sequential(model,nn.sigmoid())

model = timm.create_model("hf_hub:timm/convnextv2_large.fcmae_ft_in22k_in1k", pretrained=True)
model.stem[0] = nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
model.head.fc = nn.Linear(in_features=1536, out_features=15, bias=True).to(device)
model = model.to(device)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.00001)

num_epochs = 10
sigmoid = nn.Sigmoid()

def train(num_epochs, model, loaders):
  model.train()
  total_step = len(loaders['train'])
  min_val_loss = np.inf
  for epoch in range(num_epochs):
    train_loss = 0.0
    for i, (images,labels) in enumerate(loaders['train']):
        b_x, b_y = Variable(images), Variable(labels)
        b_y = b_y.float()
        b_x, b_y = b_x.to(device), b_y.to(device)
        output = model(b_x)
        loss = loss_func(sigmoid(output), b_y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = 0.0
    model.eval()
    for data, labels in loaders['val']:
        data, labels = data.to(device), labels.to(device)
        target = model(data)
        loss = loss_func(sigmoid(target),labels.float())
        val_loss += loss.item()
    
    if min_val_loss > val_loss:
        print(f"Validation Loss Decreased({(min_val_loss/len(loaders['val'])):.6f}--->{(val_loss/len(loaders['val'])):.6f}) \t Saving The Model")
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'saved_model_15.pth')

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {(train_loss/len(loaders['train'])):.4f}, Validation Loss: {(val_loss/len(loaders['val'])):.4f}")
      
train(num_epochs, model, loaders)

def test():
    model.eval()
    accuracy_all = []
    with torch.no_grad():
        correct = 0
        total = 0
        metric = BinaryAUROC(num_tasks=15)
        metric_sigmoid = metric = BinaryAUROC(num_tasks=15)
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            #print("output ",output)
            #output_sigmoid = torch.sigmoid(output)
            #print("output_sigmoid ",output_sigmoid)
            #final_output = (output_sigmoid>0.5).float()
            #print("Test_output size ",final_output.shape)
            #print("Test_output ",final_output)
            #print("Labels ",labels)
            #print((final_output == labels).sum().item())
            #print(float(labels.size(0)*output.shape[1]))
            #accuracy = (final_output == labels).sum().item() / float(labels.size(0)*output.shape[1])
            #accuracy_all.append(accuracy)
            #print('Test Accuracy of the model on the 1000 test images: %.2f'%accuracy)
            output_transposed = torch.transpose(output, 0, 1)
            labels_transposed = torch.transpose(labels, 0, 1)
            metric.update(output_transposed,labels_transposed)
            metric_sigmoid.update(sigmoid(output_transposed),labels_transposed)
            #print("AUC ",metric.compute())
    #print("Sum ",sum(accuracy_all))
    #print("Total ",len(accuracy_all))
    #print("Final Accuracy is ",sum(accuracy_all)/len(accuracy_all))
    print("Final AUC ",metric.compute())
    print("Final AUC Sigmoid",metric_sigmoid.compute())

test()
