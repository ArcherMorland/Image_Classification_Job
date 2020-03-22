import torch
import torch.nn as nn
import math

import os

#https://blog.csdn.net/qq_41940950/article/details/98658677

def parse_model_config(config_path):
    #print(config_path)
    parsing_result=list()
    with open(config_path, 'r') as f:
        lines=f.readlines()
        #block_open=False
        block=dict()
        for line in lines:
            line=line.replace('\n','')
            if line.strip().startswith('#') or line=='':
                continue
            if line.strip().startswith('[') and ']' in line:
                if bool(block):
                    parsing_result.append(block)
                block=dict()
                redd=line[:line.index('[')+1]
                typename=line.replace(redd,'').split(']')[0]
                block.update({'type':typename})

                if typename == 'convolutional':
                    block.update({'batch_normalize':0})
                
                    
            else:
                key,value = line.replace(' ','').split('=')
                block.update({ key:value })
        parsing_result.append(block)#add the last module
    return parsing_result


def create_modules(module_defs):
    
    hyperparameters=module_defs.pop(0)
    
    output_filters = [int(hyperparameters["channels"])]
    module_list = nn.ModuleList()

    for module_i, module_params_set in enumerate(module_defs):
        module_seq = nn.Sequential()

        #Building Convolutional Blocks
        if module_params_set['type']=='convolutional':
            
            bn = int(module_params_set['batch_normalize'])
            filters = int(module_params_set['filters'])
            kernel_size = int(module_params_set['size'])
            stride = int(module_params_set['stride'])
            pad = (kernel_size-1)//2
            
            module_seq.add_module(f'conv_{module_i}', nn.Conv2d( in_channels=output_filters[-1],
                                                                 out_channels=filters,
                                                                 kernel_size=(kernel_size, kernel_size),
                                                                 stride=stride,
                                                                 padding=pad,
                                                                 bias=not bn
                                                                ))
            if bn:
                module_seq.add_module(f'batch_norm_{module_i}', nn.BatchNorm2d(num_features=filters, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True))
            if module_params_set['activation']=='leaky':
                module_seq.add_module(f'leaky_{module_i}', nn.LeakyReLU(negative_slope=0.1, inplace=False)) 

        #Building Shortcut Blocks
        elif module_params_set['type']=='shortcut':
            #filters=output_filters[int(module_params_set['from'])]
            filters = output_filters[1:][int(module_params_set['from'])]
            module_seq.add_module(f'shortcut_{module_i}', EmptyLayer()) 
            
        #Building Route Block
        #elif module_params_set['type']=='route':


        output_filters.append(filters)
        module_list.append(module_seq)
    return hyperparameters, module_list





class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Darknet_Feature_Extractor(nn.Module):
    def __init__(self, config_path, img_size=416, num_class=1000):
        super(Darknet_Feature_Extractor,self).__init__()
        
        self.module_defs=parse_model_config(config_path)
        
        self.hyperparams, self.module_list = create_modules(self.module_defs)

        self.classifier=nn.Sequential(
            nn.Linear(75*13*13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        
    def forward(self,x):
        #x.shape#https://blog.csdn.net/weicao1990/article/details/93204452
        layer_outputs= []
        #yolo_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional']:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            layer_outputs.append(x)

        x=x.view(x.size(0), -1)
        x=self.classifier(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_class=1000):
        super(VGG16,self).__init__()
        self.feature=nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(2, 2),
        )

        self.classifier=nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        


    def forward(self, x):
        x=self.feature(x)
        x=x.view(x.size(0), -1)
        x=self.classifier(x)
        return x

#https://blog.csdn.net/lyl771857509/article/details/84175874
'''
class ResNet50(nn.Module):
    def __init__(self, num_class):
'''






if __name__ == "__main__":
    

    
   
    model_def_path=os.path.join('.','Configuration','models','Darknet_FE.cfg')

    

    model=Darknet_Feature_Extractor(model_def_path)

    print(model)



























































































