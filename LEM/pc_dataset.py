import os, random

import glob

import torch
import torch.utils.data as data

import torch
import torch.utils.data as data
import torchvision






Dataset_root=os.path.join('..','train')



def get_TraVal(Augment=True, size=0.2, stratify='target'):
    global Dataset_root

    if stratify.replace(' ','').lower() in ['target', 'y']:
        pass
    else:
        pass
    if Augment:

        #https://pytorch.org/docs/stable/torchvision/transforms.html

        transform_options = torchvision.transforms.Compose([

            torchvision.transforms.RandomCrop(size=(220, 330), padding=5),
            #size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
            #padding=4，則上下左右均填充4個pixel，若為32*32，則變成40*40
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),

            #torchvision.transforms.normalize = transforms.Normalize( mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            #these values are set for ImageNet images.
            #Implementing Normalization by using batch normal later in model.   
            ])

    else:
        transform_options = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        pass



    train_dataset=torchvision.datasets.ImageFolder(

    root=Dataset_root,
    transform=transform_options

    )


    #train_dataset.samples: check image path  #https://www.itread01.com/content/1542309028.html
    #train_dataset.class_to_idx: check class indice  #https://discuss.pytorch.org/t/how-to-know-which-image-is-at-what-index-in-torchvision-datasets-imagefolder/20808


    train_loader=data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
        )
    
    return train_loader



