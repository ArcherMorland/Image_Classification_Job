import os, random, shutil

import glob

import torch
import torch.utils.data as data

import torch
import torch.utils.data as data
import torchvision






Train_root=os.path.join('..','train')
Valid_root=os.path.join('..','valid')

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        #datas=list(map(self.turn_onehot, datas))
        batch_imgs=None
        batch_labels=None
        for i, data in enumerate(datas):
            #print(i)
            if i == 0:
                batch_imgs=data[0].view(-1, 3, 220, 330)
                batch_labels=torch.tensor([data[1]])
                #print(batch_imgs.shape)#torch.Size([3, 220, 330])
                #print(batch_labels)
            else:
                batch_imgs=torch.cat([batch_imgs, data[0].view(-1, 3, 220, 330)], 0)
                batch_labels=torch.cat([batch_labels, torch.tensor([data[1]]) ], 0)
            '''
            '''
            
            #print('datas: ',datas)
        #print(batch_imgs)
        #print(batch_labels)
        return [batch_imgs, torch.nn.functional.one_hot(batch_labels, num_classes=15)]

    

def get_TraVal(Augment=True, size=0.2, stratify='target', refresh=False):
    global Train_root, Valid_root

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

        transform_options = torchvision.transforms.Compose([

            torchvision.transforms.Resize(size=(220, 330), interpolation=2),#https://pytorch.org/docs/master/torchvision/transforms.html
            torchvision.transforms.RandomCrop(size=(220, 330), padding=5),
            #size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
            #padding=4, 則上下左右均填充4個pixel，若為32*32，則變成40*40
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),

            #torchvision.transforms.normalize = transforms.Normalize( mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            #these values are set for images of ImageNet .
            #Implementing Normalization by using batch normal later in model.   
            ])

    else:
        transform_options = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])


    #train_dataset=torchvision.datasets.ImageFolder( root=Train_root, transform=transform_options)
    train_dataset=ImageData(torchvision.datasets.ImageFolder( root=Train_root, transform=transform_options) )
    valid_dataset=torchvision.datasets.ImageFolder( root=Valid_root, 
            
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),]))


    #train_dataset.samples: check image path  #https://www.itread01.com/content/1542309028.html
    #train_dataset.class_to_idx: check class indice  #https://discuss.pytorch.org/t/how-to-know-which-image-is-at-what-index-in-torchvision-datasets-imagefolder/20808
    #print(train_dataset.class_to_idx)
    #print(valid_dataset.class_to_idx)
    
    train_loader=data.DataLoader(
        dataset=train_dataset,
        batch_size=3,
        #num_workers=4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
        )

    val_loader=data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        )
    #return train_dataset,valid_dataset
    return train_loader, val_loader


