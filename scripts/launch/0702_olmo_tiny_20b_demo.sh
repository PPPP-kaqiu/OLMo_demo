export WANDB_API_KEY=b754941545ab815dc4ca11517cc77f303864915f
MASTER_PORT=29509
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT train.py /root/paddlejob/OLMo_demo/configs/tiny/OLMo-20M.yaml --save_overwrite