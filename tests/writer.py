import argparse
import time
import torch
import torch.nn as nn
from shm_alloc_python import SHMRegion

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def main():
    parser = argparse.ArgumentParser(description="Shared memory model weight allocator writer.")
    parser.add_argument("--backend", choices=["shm", "dax"], default="shm", help="Memory backend to use")
    parser.add_argument("--path", default="/pytorch_model_weights_shm", help="POSIX shm name or DAX device path")
    args = parser.parse_args()

    use_dax = (args.backend == "dax")
    print(f"Writer: Initializing model...")
    model = SimpleModel()
    
    with torch.no_grad():
        model.fc1.weight.fill_(1.0)
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(2.0)
        model.fc2.bias.fill_(1.0)
        
    state_dict = model.state_dict()
    
    print(f"Writer: Creating shared memory region at {args.path} (Backend: {args.backend.upper()})...")
    region = SHMRegion(name=args.path, size_bytes=16*1024*1024, create=True, use_dax=use_dax)
    
    for name, tensor in state_dict.items():
        shape = list(tensor.shape)
        dtype = tensor.dtype
        element_size = tensor.element_size()
        num_elements = tensor.numel()
        nbytes = num_elements * element_size
        
        print(f"Writer: Allocating '{name}' of shape {shape} ({nbytes} bytes)...")
        obj_id, offset = region.alloc(name, nbytes)
        
        shm_tensor = region.get_tensor(offset, shape, dtype)
        shm_tensor.copy_(tensor)
        print(f"Writer: Copied weights for '{name}'. Values:\n{shm_tensor}")
        
    sync_id, sync_offset = region.alloc("sync_ready", 8)
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64)
    sync_tensor[0] = 0
    
    print("Writer: Model weights allocated and populated in shared memory.")
    print("Writer: Waiting for reader process to attach and signal done. Press Ctrl+C to terminate.")
    
    sync_tensor[0] = 1
    
    try:
        while True:
            val = sync_tensor[0].item()
            if val == 2:
                print("Writer: Reader process completed task! Exiting...")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Writer: Terminated by user.")
    finally:
        region.close(unlink=True)

if __name__ == "__main__":
    main()
