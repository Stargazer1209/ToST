
# ToST on Language Tasks


## Installation

```
pip install torch numpy transformers datasets tiktoken wandb tqdm einops
```


## Data Preparation

To prepare [OpenWebText](https://huggingface.co/datasets/openwebtext):

```sh
python data/openwebtext/prepare.py
```


## Training and Evaluation

To train TOST models (e.g. ToST_Base) after [OpenWebText](https://huggingface.co/datasets/openwebtext) is prepared:
```sh
torchrun --standalone --nproc_per_node=4 train.py /config/train_tost_base_ls.py
```

To evaluate the trained models, prepare other datasets by running:
```sh
python data/{dataset}/prepare.py
``` 

Make sure to set the correct `dataset` in `config/eval_gpt2.py`, and then run:
```sh
python train.py config/eval_gpt2.py
```

Further, to run classification tasks on Lambada and CBT, run:
```sh
python eval_lambada.py config/eval_gpt2.py
python eval_cbt.py config/eval_gpt2.py
```

## Results
After following the commands above, you should expect the following performance (in validation loss):

| model | params | OpenWebText | Lambada | Wikitext | PTB |
| ------| ------ | ----------- | ------- | -------- | --- |
| gpt2-base | 124M |  2.84 | 4.32 | 4.13 | 5.75 |
| tost-base | 110M |  3.20 | 4.98 | 4.77 | 6.39 |
| tost-medium | 304M   |  2.88 | 4.45 | 4.30 | 5.64 |
| tost-large | 655M     | 2.72 | 4.32 | 3.99 | 5.03 |


## Acknowledgements

This part of our repo is largely based on the amazing [nanoGPT](https://github.com/karpathy/nanoGPT) project.
