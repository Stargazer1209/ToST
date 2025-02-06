wandb_log = True
wandb_project = 'owt'
wandb_run_name='tost-base-ls'
out_dir = 'tost-base-ls'
always_save_checkpoint = True
# init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

batch_size = 60
block_size = 1024
gradient_accumulation_steps = 2  * 4
TSSA = True

max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-2
compile = True
