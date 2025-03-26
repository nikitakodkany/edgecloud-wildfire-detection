import cProfile
import pstats
import torch
import torchvision.models as models

def profile_function(func, *args, **kwargs):
    """Profiles a given function using cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()

def profile_pytorch_model(model, input_size):
    """Profiles a PyTorch model using torch.profiler."""
    model.eval()
    inputs = torch.randn(*input_size)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if __name__ == "__main__":
    # Example usage of cProfile
    def example_function():
        sum([i**2 for i in range(10000)])

    print("Profiling example_function with cProfile:")
    profile_function(example_function)

    # Example usage of torch.profiler
    model = models.resnet18()
    input_size = (1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    print("\nProfiling ResNet18 model with torch.profiler:")
    profile_pytorch_model(model, input_size)
