wandb_log = True
wandb_project = 'owt'
wandb_run_name='tost-medium-ls'
out_dir = 'tost-medium-ls'
always_save_checkpoint = True
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

batch_size = 30
block_size = 1024
gradient_accumulation_steps = 2  *  8
TSSA = True

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

n_layer = 24
n_head = 16
n_embd = 1024
# weight decay
weight_decay = 1e-1
compile = True
