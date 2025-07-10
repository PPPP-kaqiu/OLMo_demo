export WANDB_API_KEY=b754941545ab815dc4ca11517cc77f303864915f
export http_proxy=http://agent.baidu.com:8188
export https_proxy=http://agent.baidu.com:8188
MASTER_PORT=29509
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT train.py /root/paddlejob/OLMo_demo/scripts/launch/GPT2.yaml --save_overwrite