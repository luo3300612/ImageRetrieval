import argparse
from ImageRetrieval.backend.models import build_model
from ImageRetrieval.backend.dataset.dataset import ImageRetrievalDataset
from ImageRetrieval.backend.dataset.dataloader import build_dataloader
from tqdm import tqdm
import lmdb
import torch


def extract_features_local(model, dataloader, args):
    with torch.no_grad():
        for image in tqdm(dataloader, total=len(dataloader)):
            features = model(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image paths')
    parser.add_argument('--output', help='output image features')
    args = parser.parse_args()

    dataset = ImageRetrievalDataset(args.input)
    dataloader = build_dataloader(dataset, args)
    model = build_model(args).cuda()

    extract_features_local(model, dataloader, args)
