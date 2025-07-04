export WANDB_API_KEY=b754941545ab815dc4ca11517cc77f303864915f
torchrun --nproc_per_node=8 train.py /root/paddlejob/OLMo/configs/tiny/OLMo-20M.yaml