MASTER_PORT=29598
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT stream_dataset_dist.py