import argparse
from ImageRetrieval.backend.models import build_model
from ImageRetrieval.backend.dataset.dataset import ImageRetrievalDataset
from ImageRetrieval.backend.dataset.dataloader import build_dataloader
from tqdm import tqdm
import lmdb
import torch
import numpy as np


def extract_features_local(model, dataloader, args):
    all_features = []
    with torch.no_grad():
        for image in tqdm(dataloader, total=len(dataloader)):
            image = image.cuda()
            features = model(image)
            features = features.cpu()
            all_features.append(features)
    return torch.cat(all_features, dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image paths')
    parser.add_argument('--output', help='output image features')
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--use_imagenet_pretrained', action='store_true')
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()

    dataset = ImageRetrievalDataset(args.input)
    dataloader = build_dataloader(dataset, args)
    model = build_model(args).cuda()

    features = extract_features_local(model, dataloader, args)
    features = features.numpy()
    np.save(args.output, features)
    # print(features)
    print('Done')
