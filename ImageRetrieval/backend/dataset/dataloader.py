from torch.utils.data import DataLoader


def build_dataloader(dataset, args):
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    return dataloader
