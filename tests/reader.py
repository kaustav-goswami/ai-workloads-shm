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
    parser = argparse.ArgumentParser(description="Shared memory model weight allocator reader.")
    parser.add_argument("--backend", choices=["shm", "dax"], default="shm", help="Memory backend to use")
    parser.add_argument("--path", default="/pytorch_model_weights_shm", help="POSIX shm name or DAX device path")
    args = parser.parse_args()

    use_dax = (args.backend == "dax")
    print(f"Reader: Attaching to shared memory region at {args.path} (Backend: {args.backend.upper()})...")
    
    region = SHMRegion(name=args.path, create=False, use_dax=use_dax)
    
    print("Reader: Locating sync variable...")
    sync_id, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64)
    
    print("Reader: Waiting for writer process to populate weights...")
    while sync_tensor[0].item() != 1:
        time.sleep(0.1)
        
    print("Reader: Setup is ready. Attaching model parameters...")
    model = SimpleModel()
    
    param_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    shm_state_dict = {}
    for name in param_names:
        local_param = model.state_dict()[name]
        shape = list(local_param.shape)
        dtype = local_param.dtype
        
        obj_id, offset = region.lookup_by_name(name)
        shm_tensor = region.get_tensor(offset, shape, dtype)
        shm_state_dict[name] = shm_tensor
        print(f"Reader: Attached parameter '{name}' from shared memory. Values:\n{shm_tensor}")
        
    model.load_state_dict(shm_state_dict)
    print("Reader: Model parameters attached successfully!")
    
    print("Reader: Running forward pass validation...")
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    print(f"Reader: Input tensor x: {x}")
    
    with torch.no_grad():
        out = model(x)
        
    print(f"Reader: Output tensor: {out}")
    
    expected = torch.tensor([[37.0, 37.0]])
    print(f"Reader: Expected output: {expected}")
    
    if torch.allclose(out, expected):
        print("Reader: SUCCESS! Shared memory weights loaded and verified correctly.")
    else:
        print("Reader: FAILURE! Output does not match expected values.")
        
    sync_tensor[0] = 2
    region.close(unlink=False)

if __name__ == "__main__":
    main()
