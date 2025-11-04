# import sys
# sys.path.append(r"D:\Codes\PyTorch_nn")
# from model.resnet import resnet18, resnet50

# model_dict = {
#     'resnet18': resnet18,
#     'resnet50': resnet50,
# }

# def create_model(model_name, num_classes):   
#     return model_dict[model_name](num_classes = num_classes)

import sys
sys.path.append(r"\Users\kindle\Desktop\file\project\pytorch_nn")
from notebook.VIT import vit_base_patch16_224

model_dict = {
    'vit' : vit_base_patch16_224,
}

def create_model(model_name, num_classes):   
    return model_dict[model_name](num_classes = num_classes)