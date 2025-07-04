import os
from streaming import StreamingDataset

# Directory containing the .mds shards
remote_dir = '/root/paddlejob/data/redpajama/for_ft/arxiv/' # Or the full path to the 'arxiv' directory

# Create a StreamingDataset instance
dataset = StreamingDataset(remote=remote_dir, batch_size=1, local="/root/paddlejob/data/redpajama/cache/")

# Access a specific sample (e.g., the first one)
sample = dataset[0]

# Print the sample content
print(sample)

# You can also iterate through a few samples
print("\n--- Iterating through the first 5 samples ---")
for i, sample in enumerate(dataset):
    if i >= 5:
        break
    print(f"\n--- Sample {i} ---")
    # The content is usually in a dictionary. The key might be 'text' or something similar.
    # You may need to inspect the first sample to see the exact key.
    print(sample.get('text', 'No text key found'))