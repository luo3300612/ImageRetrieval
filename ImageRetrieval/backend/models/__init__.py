import torch
from .resnet import resnet18


def build_model(args):
    assert args.use_imagenet_pretrained or args.pretrained_weights != '', 'please specify pretrained weights'
    # 同时需要实现模型加载
    if args.model == 'resnet18':
        model = resnet18(pretrained=args.use_imagenet_pretrained)
    else:
        raise NotImplementedError(f'Not implemented for model {args.model}')

    if not args.use_imagenet_pretrained:
        state_dict = torch.load(args.pretrained_weights)
        model.load_state_dict(state_dict)

    return model
