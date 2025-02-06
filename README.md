# Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction
This repo contains the implementation for ToST (Token Statistics Transformer), a linear-time architecture derived via algorithmic unrolling.

<div align="center">
  <image src="assets/tost_arch.jpg" width="840px" />
  <p></p>
</div>

<div align="center">
<a href="https://arxiv.org/pdf/2412.17810" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ToST-red?logo=arxiv" height="20" />
</a>
<a href="https://robinwu218.github.io/ToST" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-ToST-blue.svg" height="20" />
</a>
</div>

## Updates
- [02/05/25] Code for ToST on vision, language tasks is released!
- [01/22/25] Accepted to ICLR 2025!

## Usage

We have organized the implementation for vision and language tasks into the respective `tost_vision` and `tost_lang` directories. Please follow the instuctions within them. We recommend useing separate environments for these two implementations.

## Citation

If you find this project helpful for your research and applications, please consider cite our work:
```bibtex
@article{wu2024token,
  title={Token Statistics Transformer: Linear-Time Attention via Variational Rate Reduction},
  author={Wu, Ziyang and Ding, Tianjiao and Lu, Yifu and Pai, Druv and Zhang, Jingyuan and Wang, Weida and Yu, Yaodong and Ma, Yi and Haeffele, Benjamin D},
  journal={arXiv preprint arXiv:2412.17810},
  year={2024}
}
```

## Acknowledgements
- [XCiT: Cross-Covariance Image Transformer](https://github.com/facebookresearch/xcit): the code for vision is largely based on this repo.
- [nanogpt](https://github.com/karpathy/nanoGPT): the code for language is mostly based on this repo.



