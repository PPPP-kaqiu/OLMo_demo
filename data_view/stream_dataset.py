import numpy as np
from torch.utils.data import DataLoader
from streaming import StreamingDataset, Stream
from transformers import AutoTokenizer
# Local working dir where dataset is cached during operation
local = '/root/paddlejob/redpajama/for_ft/arxiv'
tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/shakechen/Llama-2-7b-chat-hf")
streams = [
  Stream(local='/root/paddlejob/redpajama/for_ft/arxiv/', proportion=0.007),
  Stream(local='/root/paddlejob/redpajama/for_ft/book/', proportion=0.091),
  Stream(local='/root/paddlejob/redpajama/for_ft/c4-rp/', proportion=0.492),
  Stream(local='/root/paddlejob/redpajama/for_ft/cc/', proportion=0.361),
  Stream(local='/root/paddlejob/redpajama/for_ft/github/', proportion=0.008),
  Stream(local='/root/paddlejob/redpajama/for_ft/stackexchange/', proportion=0.01),
  Stream(local='/root/paddlejob/redpajama/for_ft/wiki/', proportion=0.031)
]

dataset = StreamingDataset(
  streams=streams
)

# Let's see what is in sample #1337...
sample = dataset[1337]
token_bytes = sample['tokens']
token_ids = np.frombuffer(token_bytes, dtype=np.uint16)
detokenize_token = tokenizer.decode(token_ids, skip_special_tokens=False)
print(token_ids)
print(detokenize_token)
# Create PyTorch DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    drop_last=train_config.data.drop_last,
    collate_fn=collator,
    num_workers=train_config.data.num_workers,
    pin_memory=train_config.data.pin_memory,
    prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
    persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
    timeout=train_config.data.timeout,
)
for batch in dataloader:
    import ipdb;ipdb.set_trace()
    pass