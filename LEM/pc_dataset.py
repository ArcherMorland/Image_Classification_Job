import os, random, shutil

import glob

import torch
import torch.utils.data as data

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets






Train_root=os.path.join('..','train')
Valid_root=os.path.join('..','valid')

ImgH=224  
ImgW=224 

def return_all():
    global Valid_root

    imgs_Val=glob.glob( os.path.join(Valid_root, '*', '*.*') )
    
    while imgs_Val : 
        p=imgs_Val.pop()
        shutil.move(p , os.path.dirname(p.replace('valid', 'train') ))

def train_val_split(split_size):
    global Train_root, Valid_root

    valid_dirs=list(map(lambda x:x[0].replace('train', 'valid'), os.walk( Train_root ) ))
    valid_dirs.pop(0)#remove first unused path=>'../valid'
    for i, vd in enumerate(valid_dirs):
        if not os.path.exists( vd ):
            os.makedirs( vd )    
        pics_single_class = glob.glob(os.path.join(vd.replace('valid','train'), '*')  )    
        selection=random.sample(pics_single_class, round(len(pics_single_class)*split_size))    
        while selection: shutil.move( selection.pop(), vd )
        #https://stackoverflow.com/questions/47722712/use-map-for-functions-that-does-not-return-a-value
        #https://stackoverflow.com/questions/1080026/is-there-a-map-without-result-in-python



class ImageData(data.Dataset):
    def __init__(self,data):
        self.data=data
        self.class_to_index = data.class_to_idx#check class indice  #https://discuss.pytorch.org/t/how-to-know-which-image-is-at-what-index-in-torchvision-datasets-imagefolder/20808
        self.num_classes=len(self.class_to_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
   
    def collate_fn(self, datas):
        #datas:[(tensor[3, 220, 330],0),(tensor[3, 220, 330],9),(tensor[3, 220, 330],14)...]
        batch_imgs=None
        batch_labels=None
        batch_classname=[]
        for i, data in enumerate(datas):
             
            batch_classname.extend([cname for cname, idx in self.class_to_index.items() if idx == data[1]])
            if i == 0:
                batch_imgs=data[0].view(-1, 3, ImgH, ImgW)# [channels, height, width]
                batch_labels=torch.tensor([data[1]])
                #print(batch_imgs.shape)#torch.Size([3, 220, 330])
                #print(batch_labels)
            else:
                batch_imgs=torch.cat([batch_imgs, data[0].view(-1, 3, ImgH, ImgW)], 0)
                batch_labels=torch.cat([batch_labels, torch.tensor([data[1]]) ], 0)
            
        batch_labels_ls=torch.nn.functional.one_hot(batch_labels, num_classes=self.num_classes).tolist()  
        batch_labels_ft= torch.FloatTensor(batch_labels_ls)
        #print("batch_classname:",batch_classname)
        return [batch_imgs, batch_labels_ft, batch_classname]
    

def get_TraVal(Augment=True, size=0.2, stratify='target', refresh=False, target_onehot=True):
    global Train_root, Valid_root, ImgH, ImgW

    if not os.path.exists( os.path.join('..','valid') ):
        os.makedirs( os.path.join('..','valid') )

    if refresh and stratify.replace(' ','').lower() in {'target', 'y'}:
        return_all()
        train_val_split(size)
        
    elif refresh:
        return_all()
        pass



    if Augment:

        #https://pytorch.org/docs/stable/torchvision/transforms.html

        transforms_options = torchvision.transforms.Compose([

            torchvision.transforms.Resize(size=(ImgH, ImgW), interpolation=2),#(h, w)
            #https://pytorch.org/docs/master/torchvision/transforms.html
            torchvision.transforms.RandomCrop(size=(ImgH, ImgW), padding=5),#(h, w)
            #size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
            #padding=4, 則上下左右均填充4個pixel，若為32*32，則變成40*40
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),

            #torchvision.transforms.normalize = transforms.Normalize( mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            #these values are set for images of ImageNet .
            #Implementing Normalization by using batch normal later in model.   
            ])

    else:
        transforms_options = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(ImgH, ImgW), interpolation=2),#(h, w)
            torchvision.transforms.ToTensor(),
            ])


    #train_dataset=torchvision.datasets.ImageFolder( root=Train_root, transform=transform_options)
    if target_onehot:
        train_dataset=ImageData(torchvision.datasets.ImageFolder( root=Train_root, transform=transforms_options) )
        valid_dataset=ImageData(torchvision.datasets.ImageFolder( root=Valid_root, transform=transforms.Compose([ transforms.Resize(size=(ImgH, ImgW), interpolation=2),#(h, w) 
                                                                                                                  transforms.ToTensor(),]) ))


        train_loader=data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn  )

        val_loader=data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn  )                                 
    else:
        train_dataset=torchvision.datasets.ImageFolder( root=Train_root, transform=transforms_options)
        valid_dataset=torchvision.datasets.ImageFolder( root=Valid_root, transform=transforms.Compose([ transforms.Resize(size=(ImgH, ImgW), interpolation=2),#(h, w) 
                                                                                                        transforms.ToTensor(),]) )
        #train_dataset.samples: check image path  #https://www.itread01.com/content/1542309028.html
        #train_dataset.class_to_idx: check class indice  #https://discuss.pytorch.org/t/how-to-know-which-image-is-at-what-index-in-torchvision-datasets-imagefolder/20808
        train_loader=data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        )

        val_loader=data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        )        
    
    
    #return train_dataset,valid_dataset
    return train_loader, val_loader


