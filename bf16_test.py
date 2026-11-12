import sys
sys.path.insert(0, '.')
from src.storage.safetensors_ranges import parse_safetensors
import torch

info = parse_safetensors('models/tinyllama.safetensors')
t = info['tensors']['model.layers.0.self_attn.q_proj.weight']
print('meta', t)
print('abs_start', t['abs_start'], 'abs_end', t['abs_end'], 'length', t['length'], 'shape', t['shape'])

raw = open('models/tinyllama.safetensors', 'rb').read()
header_len = int.from_bytes(raw[0:8], 'little')
data_start = 8 + header_len
data = raw[data_start:]
seg = data[t['abs_start'] : t['abs_end']]
print('seg bytes', len(seg), 'expected', t['length'])

# method A: native bf16
a = torch.frombuffer(memoryview(seg), dtype=torch.uint16).view(torch.bfloat16).contiguous().view(torch.float32).tolist()
print('native bf16  sample:', a[:5])
# method B: f32 zero-lower
b = torch.frombuffer(memoryview(seg), dtype=torch.uint8).view(torch.int32).view(torch.float32).tolist()
print('f32  first5:', b[:5])
