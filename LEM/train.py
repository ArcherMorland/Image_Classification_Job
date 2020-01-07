import pc_dataset as PC_Dataset
from pc_dataset import get_TraVal

import os,time, random
import glob

import PIL,cv2
import matplotlib.pyplot as plt


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







traindata_loader=get_TraVal()


for batch_idx, (data, target) in enumerate(traindata_loader):
    
    
    #show(data[0])
    #print(batch_idx)
    
    if batch_idx>-1:
        break
















#'*.jpg'

size=0.2
if not os.path.exists( os.path.join('..','valid') ):
    os.makedirs( os.path.join('..','valid') )

valid_dirs=list(map(lambda x:x[0].replace('train', 'valid'), os.walk(os.path.join('..','train',)) ))
valid_dirs.pop(0)#remove first unused path=>'../valid'
for i, vd in enumerate(valid_dirs):
    if not os.path.exists( vd ):
        os.makedirs( vd )
    #print(os.path.dirname(vd) )
    pics_single_class = glob.glob(os.path.join(vd.replace('valid','train'), '*')  )
    
    print(pics_single_class)
    selection=random.sample(pics_single_class, round(len(pics_single_class)*size))
    print(selection)
    if i>-1:
        break

#print(glob.glob( os.path.join('..','train','*',)))

