#!/usr/bin/env python
# coding: utf-8


import os, glob, tqdm, json
import cv2
from torch.utils.data import Dataset, DataLoader

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data_dir = './data/food-11/'
model_name, size = 'resnext50_32x4d', (224, 224)
batch_size = 24
n_epoch = 1

train_img_paths = glob.glob(os.path.join(data_dir, 'training', '*.jpg'))
valid_img_paths = glob.glob(os.path.join(data_dir, 'validation', '*.jpg'))
test_img_paths = glob.glob(os.path.join(data_dir, 'testing', '*.jpg'))


class FoodDataset(Dataset):
    def __init__(self, img_paths, size):
        self.img_paths = img_paths
        self.path_label_map = { path: int(path.split('/')[-1].split('.')[0].split('_')[0]) for path in img_paths}
        
        self.size = size
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        x = self._getImage(path)
        y = self.path_label_map[path]
        return x, y

    def _getImage(self, path: str):
        img = self._loadImage(path)
        img = self._preprocess(img)
        return img
        
    def _loadImage(self, path: str):
        img = cv2.imread(path)
        return img

    def _preprocess(self, img: 'numpy.array'):
        img = cv2.resize(img, self.size, interpolation = cv2.INTER_AREA)
        
        img = img / 255
        
#         # min-max normalization (by channel)
#         g_max, g_min = img[:, :, 0].max(), img[:, :, 0].min()
#         b_max, b_min = img[:, :, 1].max(), img[:, :, 1].min()
#         r_max, r_min = img[:, :, 2].max(), img[:, :, 2].min()

#         img[:, :, 0] = (img[:, :, 0] - g_min) / (g_max - g_min)
#         img[:, :, 1] = (img[:, :, 1] - b_min) / (b_max - b_min)
#         img[:, :, 2] = (img[:, :, 2] - r_min) / (r_max - r_min)
        
        img = img.transpose((2,0,1))
        return img
    
    
train_dataset = FoodDataset(train_img_paths, size)
valid_dataset = FoodDataset(valid_img_paths, size)
test_dataset = FoodDataset(test_img_paths, size)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self, model_name = 'resnext50_32x4d'):
        super(Net, self).__init__()
        backbone, output_dim = self._createBackbone(model_name)
        
        self.backbone = backbone
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(output_dim, 11)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
        
    
    def _createBackbone(self, model_name):
        backbone = timm.create_model(model_name, pretrained=True)
        output_dim = None
        
        if model_name == 'resnext50_32x4d':
            backbone.fc = nn.Identity()
            backbone.global_pool = nn.Identity()
            output_dim = 2048
        elif model_name == 'efficientnet_b3':
            backbone.classifier = nn.Identity()
            backbone.global_pool = nn.Identity()
            output_dim = 1536
        else:
            raise ValueError('[Error] Unexpected model name: {}'.format(model_name))
            
        return backbone, output_dim
    
model = Net(model_name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)


best_accuracy = 0
history = {
    'train_loss': [],
    'valid_loss': [],
    'valid_acc': [],
}

print('[INFO] Train start')
for epoch in range(n_epoch):
    
    train_running_loss, train_n_time = 0.0, 0
    valid_running_loss, valid_n_time = 0.0, 0
    n_accuracy, n_valid = 0, 0 
    
    model.train()
    for inputs, labels in tqdm.tqdm(train_dataloader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        
        preds = model(inputs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        train_n_time += 1
    
    
    model.eval()
    for inputs, labels in  tqdm.tqdm(valid_dataloader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
        
        preds = model(inputs)
        loss = criterion(preds, labels)
        
        valid_running_loss += loss.item()
        valid_n_time += 1
        
        n_accuracy += (preds.argmax(-1) == labels).sum().item()
        n_valid += inputs.shape[0]
    
    train_loss = train_running_loss / train_n_time
    valid_loss = valid_running_loss / valid_n_time
    accuracy = n_accuracy / n_valid
    print('epoch: {}, train_loss: {:.6f}, valid_loss: {:.6f}, valid_acc: {:.2f}%' \
          .format(epoch, train_loss, valid_loss, 100 * accuracy))
    
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['valid_acc'].append(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), model_name + '.weights.pth')
        print('[INFO] New highest accuracy: {:.2f}%, weights saved.'.format(100 * best_accuracy))
        
with open(model_name + '.history.json', 'w') as file:
    json.dump(history, file)
    
print('[INFO] Train finish')