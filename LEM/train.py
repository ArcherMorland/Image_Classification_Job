import pc_dataset as PC_Dataset
from pc_dataset import get_TraVal, return_all
from networks import VGG16

import os, time, json

import PIL,cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#A small tool for evaluating processing time of a function
def FunctionTimer(f,args):    
   
    if not callable(f):
        raise TypeError("It is not callable!!")
    t1=time.time()
    result=f(*args)
    t2=time.time()
    print("{n} takes : {t:.10f} s".format(n=f.__name__ ,t=t2-t1))
   
    return result
def show(imgT):
    plt.imshow(imgT.permute(1,2,0))#PyTorch Tensors ("Image tensors") are channel first, so to use them with matplotlib user need to reshape it
                                   #https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch/53633017
                                   #像素顺序是RGB(critical reference!!!!): https://www.jianshu.com/p/c0ba27e392ff
                                   #to gray_scale:https://stackoverflow.com/questions/52439364/how-to-convert-rgb-images-to-grayscale-in-pytorch-dataloader
    plt.show()

#def compute_acc(pred,lab):
 #   return




traindata_loader, valdata_loader=get_TraVal( refresh=False, target_onehot=False)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#tol=20

best_valACC=0
tol_list=list()
tol=20
def _run_learning(training=True,epoch=0):
    global best_valACC, tol_list, tol
    
    loss_total=0
    
    if training:
        stage='Train'
        model.train(True)
        dataloader = traindata_loader

    else:
        stage='Valid'
        model.train(False)
        dataloader = valdata_loader
        val_hits=0


    for batch_idx, ( data, target) in enumerate(dataloader):
        #print(data.shape)#torch.Size([32, 3, 220, 330])

        batch_data=data.to(device)
        batch_label=target.to(device)
        
        prediction = model(batch_data)

        #loss = criterion(torch.sigmoid(prediction), batch_label)
        loss = criterion(prediction, batch_label)#多分類用的交叉熵損失函數，用這個loss前不需要加Softmax層 =>https://blog.csdn.net/zhangxb35/article/details/72464152
        #loss = criterion(F.softmax(prediction,dim=1), batch_label)
        
        #loss = F.binary_cross_entropy_with_logits(prediction, batch_label)
        
        
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            #print(F.softmax(prediction,dim=1))
            batch_pred_mat=F.softmax(prediction,dim=1)
            batch_pred_scalar=batch_pred_mat.max(-1)[1]

            val_hits+=sum([1 if batch_pred_scalar[i]==batch_label[i] else 0 for i in range(len(batch_pred_scalar))])
            print('\n')
            print("batch_pred_scalar :",batch_pred_scalar)
            print("target : ",target)
            print('\n')
            
        loss_total+=loss.item()
    
    if not training:
        
        current_valACC=round(val_hits/len(dataloader.dataset),4)

        print("valACC", current_valACC )

        if current_valACC > best_valACC:
            best_valACC=current_valACC
            tol_list=list()
            #save model =>  https://blog.csdn.net/u012436149/article/details/68948816
        else:
            #secd_valACC
            tol_list.append(current_valACC)
            if len(tol_list)>tol:
                return False
        
        print(epoch, "stage  :",stage,", total loss: ",loss_total,',   device:',device, ",  current_valACC: ", current_valACC,"  best_valACC : ",best_valACC,"  tol_list:",tol_list)
    
    
    return True




model = VGG16( num_class=15 )
#https://www.okcode.net/article/87118
criterion = nn.CrossEntropyLoss()
'''
https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216

CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices:

The input is expected to contain scores for each class.
input has to be a 2D Tensor of size (minibatch, C).
This criterion expects a class index (0 to C-1) as the target for each value of a 1D tensor of size minibatch
'''

#criterion = nn.BCELoss()

optimizer=torch.optim.Adam(model.parameters(), lr=9e-6)



max_epoch=100
model.to(device) 


for ep in range(max_epoch):

    res_T=_run_learning(training=True,epoch=ep)
    res_V=_run_learning(training=False,epoch=ep)
    if not res_V:
        break



































