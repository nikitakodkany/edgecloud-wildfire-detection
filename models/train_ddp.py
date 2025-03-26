import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from unet_model import UNet  # Ensure you have a UNet model defined in unet_model.py

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, data_dir, epochs=5, batch_size=32):
    setup(rank, world_size)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = UNet()
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        for images, masks in dataloader:
            images, masks = images.to(rank), masks.to(rank)
            outputs = ddp_model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    data_dir = "/path/to/dataset"
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, data_dir), nprocs=world_size, join=True)