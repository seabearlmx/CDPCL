"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
import torch.nn as nn
import torch
from config import cfg

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer = layer(in_channels)
    return normalization_layer


def freeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = False

def unfreeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = True

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_() #original



def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    # print('net_state_dict')
    # print(net_state_dict.keys())
    # print('loaded_dict')
    # print(loaded_dict.keys())
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def forgiving_state_restore_vgg(net, loaded_dict):
    # 当前网络的参数
    model_dict = net.state_dict()
    # print('model_dict')
    # for k, v in model_dict.items():
    #     print(k, v.shape)
    # 加载vgg模型
    load_dict = loaded_dict
    # print('load_dict')
    # for k, v in load_dict.items():
    #     print(k, v.shape)
    key = list(model_dict.keys())
    name = list(load_dict.keys())
    weights = list(load_dict.values())

    t = 0
    new_loaded_dict = {}
    for i in range(len(weights)):
        # 不加载最后的全连接层
        if 'classifier' in name[i]:
            break
        # 当前模型使用BN层多一个num_batches_tracked，但是加载的模型中没有，因此需要跳过
        if 'weight' in key[i + t] and len(model_dict[key[i + t]].size()) == 1:
            t += 1
        if 'running_mean' in key[i + t]:
            t += 1
        if 'running_var' in key[i + t]:
            t += 1
        if 'num_batches_tracked' in key[i + t]:
            t += 1
        # print(model_dict[key[i + t]].size())
        # print(weights[i].size())
        if model_dict[key[i + t]].size() == weights[i].size():
            # print('right')
            print(key[i + t])
            print(name[i])
            # print(model_dict[key[i + t]].size())
            # print(weights[i].size())
            # print(t)
            new_loaded_dict[key[i + t]] = weights[i]
        else:
            print('skip')

    model_dict.update(new_loaded_dict)
    net.load_state_dict(model_dict)
    return net

