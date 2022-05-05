#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# # Helper function

# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # image  shape = [batch_size, 3, H, W]
                labels = labels.to(device)  #

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                

            else: # phase == 'train'
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history, train_loss, train_acc


# In[ ]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# In[ ]:


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features  # number of output/last-layer feature
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# # Finetuning AlexNet

# ## reshape

# In[ ]:


model_name = "alexnet"  

num_classes = 3  # Number of classes in the dataset;     left: 0; right: 1; virtical: 2; 
feature_extract = True  # Flag for feature extracting. When False, we finetune the whole model,
                        #   when True we only update the reshaped layer params
batch_size = 10  # Batch size for training (change depending on how much memory you have)
num_epochs = 30  # Number of epochs to train for


# Initialize the model for this run
grating_orientation_discrimination, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(grating_orientation_discrimination)  # Print the model we just instantiated


# ## optimize the last layer

# In[ ]:


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
grating_orientation_discrimination = grating_orientation_discrimination.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

params_to_update = grating_orientation_discrimination.parameters()

print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name,param in grating_orientation_discrimination.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in grating_orientation_discrimination.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001/10, momentum=0.9)  # ft: finetuning


# ## train dataset

# In[ ]:


# to the ImageFolder structure
data_dir = "NMA project/test/gratings"

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}


# ## train & evaluate

# In[ ]:


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
grating_orientation_discrimination, hist, train_loss, train_acc = train_model(grating_orientation_discrimination, dataloaders_dict, 
                             criterion, optimizer_ft, num_epochs=num_epochs, 
                             is_inception=(model_name=="inception"))


# In[ ]:


plt.figure()
plt.subplot(1, 2, 1)
plt.plot(train_loss)

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='train_acc')
plt.plot(hist, label='test_acc')
plt.legend()


# ## save model

# In[ ]:


torch.save(grating_orientation_discrimination, 'best_alexnet_model.pkl')  # save a model after trained
torch.save(grating_orientation_discrimination.state_dict(), 'best_alexnet_params.pkl') # save the weights of the model


# # Test

# ## test dataset

# In[ ]:


# to the ImageFolder structure
data_dir = "NMA project/test/gratings"

data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Create test datasets
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transform)


# ## output

# In[ ]:


input_batch = torch.cat([test_dataset[k][0].unsqueeze(0) for k in range(len(test_dataset))], dim=0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    
with torch.no_grad():
    output = grating_orientation_discrimination(input_batch)


# In[ ]:


p_out = torch.nn.functional.softmax(output, dim=1)
preds = p_out.argmax(dim=1).numpy()
labels = np.array([test_dataset[k][1] for k in range(len(test_dataset))])  


# # Figure

# In[ ]:


import pandas as pd
import seaborn as sns
sns.set_context('talk')


# In[ ]:


df = pd.DataFrame({'label': labels, 'pred': preds, 'correct': preds==labels})
df['label'] = df.label.map({0: 'left', 1: 'right', 2: 'vertical'})
df['pred'] = df.pred.map({0: 'left', 1: 'right', 2: 'vertical'})

sns.barplot(x='label', y='correct', data=df)
df.groupby(['label', 'pred']).count()

