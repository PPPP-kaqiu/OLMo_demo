import numpy as np
from torch.utils.data import DataLoader
from streaming import StreamingDataset, Stream
from transformers import AutoTokenizer
from olmo.data.collator_stream import DataCollator_Stream

# Local working dir where dataset is cached during operation
local = '/root/paddlejob/redpajama/for_ft/arxiv'
tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/shakechen/Llama-2-7b-chat-hf")
# streams = [
#   Stream(local='/root/paddlejob/redpajama/for_ft/arxiv/', proportion=0.007),
#   Stream(local='/root/paddlejob/redpajama/for_ft/book/', proportion=0.091),
#   Stream(local='/root/paddlejob/redpajama/for_ft/c4-rp/', proportion=0.492),
#   Stream(local='/root/paddlejob/redpajama/for_ft/cc/', proportion=0.361),
#   Stream(local='/root/paddlejob/redpajama/for_ft/github/', proportion=0.008),
#   Stream(local='/root/paddlejob/redpajama/for_ft/stackexchange/', proportion=0.01),
#   Stream(local='/root/paddlejob/redpajama/for_ft/wiki/', proportion=0.031)
# ]
streams = [
  Stream(local='/root/paddlejob/data/stackexchange/', proportion=1.0),
]

dataset = StreamingDataset(
  streams=streams,
  batch_size=128
)
def visualize_document(idx):
  sample = dataset[idx]
  token_bytes = sample['tokens']
  token_ids = np.frombuffer(token_bytes, dtype=np.uint16)
  detokenize_token = tokenizer.decode(token_ids, skip_special_tokens=False)
  print(token_ids)
  print(detokenize_token)
  
collator = DataCollator_Stream()
dataloader = DataLoader(
    dataset,
    batch_size=128,
    drop_last=True,
    collate_fn=collator,
    num_workers=32,
    pin_memory=True,
    prefetch_factor=8,
    persistent_workers=True,
    timeout=0,
)
for batch in dataloader:
    import ipdb;ipdb.set_trace()
    pass