from torch.utils.data import DataLoader

def build_dataloader(dataset, batch_size, workers, train=True, shuffle=True,**kwargs):
    data_loader = DataLoader(dataset,batch_size=batch_size,num_workers=workers,pin_memory=False,shuffle=True,**kwargs)

    return data_loader

