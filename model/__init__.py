
import sys
sys.path.append(r"/Users/kindle/Desktop/file/project/pytorch_nn")
from model.cnn import simple_cnn
from model.resnet import resnet18, resnet50
from model.vit_simple import vit_base_patch16_224

model_dict = {
    'simplecnn': simple_cnn,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'vit' : vit_base_patch16_224
}

def create_model(model_name, num_classes):   
    return model_dict[model_name](num_classes = num_classes)