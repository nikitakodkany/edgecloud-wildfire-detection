import torch
import torch.autograd.profiler as profiler

def profile_model(model, input_tensor):
    with profiler.profile(use_cuda=True) as prof:
        with torch.no_grad():
            model(input_tensor)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
