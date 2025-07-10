import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from streaming import StreamingDataset, Stream
from transformers import AutoTokenizer

if dist.is_available() and "WORLD_SIZE" in os.environ:
    dist.init_process_group("nccl")  # 使用 NCCL 后端
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # 将当前设备设置为该进程对应的 GPU
    torch.cuda.set_device(rank)
    print(f"分布式训练已启动：总进程数 {world_size}, 当前进程 Rank {rank}")
else:
    # 如果不是分布式环境，则设置为单卡模式
    world_size = 1
    rank = 0
    print("单卡模式运行")

base_seed = 42
seed = base_seed + rank

device_train_batch_size = 128 # 这是每个 GPU 上的 batch_size
num_workers = 32

local_cache_dir = '/root/paddlejob/data/cache'

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/shakechen/Llama-2-7b-chat-hf")
streams = [
  Stream(local='/root/paddlejob/data/stackexchange/', proportion=1.0),
]

# StreamingDataset 会自动利用 rank 和 world_size 来分发数据
dataset = StreamingDataset(
  streams=streams,
  batch_size=device_train_batch_size,
  shuffle=True,          # 开启 shuffle
  shuffle_algo='py1s',   # 推荐的 shuffle 算法
  shuffle_seed=seed,     # 使用我们为每个 rank 设置的种子
  predownload=1_000,     # 预下载的样本数
  cache_limit='100gb'    # 缓存大小限制
)

def collate_and_convert(batch):
    all_tokens = []
    for sample in batch:
        token_bytes = sample['tokens']
        # 将 bytes 转换为 uint16 的 numpy 数组，再转为 torch tensor
        token_tensor = torch.from_numpy(np.frombuffer(token_bytes, dtype=np.uint16).copy())
        all_tokens.append(token_tensor)
    # 将 list of tensors 堆叠成一个 batch tensor
    return {"input_ids": torch.stack(all_tokens)}


dataloader = DataLoader(
    dataset,
    batch_size=device_train_batch_size,
    collate_fn=collate_and_convert, # 使用自定义的 collate 函数
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=8 if num_workers > 0 else None,
    persistent_workers=True if num_workers > 0 else False,
    timeout=0,
)

if rank == 0:
    print("开始迭代数据...")

# 迭代器会自动处理每个 epoch 的数据分发
for i, batch in enumerate(dataloader):
    # 打印一些信息来验证
    if i < 2: # 只打印前几个 batch
        input_ids = batch['input_ids']
        if rank == 0:
            print(f"\nBatch {i+1}:")
            print(f"  - Shape: {input_ids.shape}")
            print(f"  - Device: {input_ids.device}")
            print(f"  - dtype: {input_ids.dtype}")
            print(f"  - Input: {input_ids}")

    if i >= 4:
        break

if rank == 0:
    import ipdb;ipdb.set_trace()
    print("\n数据加载和迭代演示完成。")