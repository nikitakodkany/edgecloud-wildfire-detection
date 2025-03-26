import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from unet import UNet

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform)  # replace with actual fire dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=4)

    model = UNet().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(5):
        for img, _ in loader:
            img = img.cuda(rank)
            out = ddp_model(img)
            loss = loss_fn(out, torch.rand_like(out))  # dummy target
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
