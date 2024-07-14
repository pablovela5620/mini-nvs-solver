# Mini-NVS-Solver
A minimal implementation of [NVS-Solver](https://github.com/ZHU-Zhiyu/NVS_Solver). Using [Rerun](https://rerun.io/) viewer, [Pixi](http://pixi.sh/) and [Gradio](https://www.gradio.app/) for easy use
<p align="center">
  <img src="media/mini-nvs-solver.gif" alt="example output" width="480" />
</p>


## Installation
Easily installable via [Pixi](https://pixi.sh/latest/).
```bash
git clone https://github.com/pablovela5620/mini-nvs-solver
cd mini-nvs-solver
pixi run app
```

## Demo
Hosted Demos can be found on huggingface spaces

TODO
<a href='https://huggingface.co/spaces/pablovela5620/depth-compare'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>


To run the gradio frontend
```bash
pixi run app
```

To see all available tasks
```bash
pixi task list
```

## Acknowledgements
Thanks to the authors for the great work!

[NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer
](https://github.com/ZHU-Zhiyu/NVS_Solver/tree/main)
```bibtex
@article{you2024nvs,
title={NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer},
author={You, Meng and Zhu, Zhiyu and Liu, Hui and Hou, Junhui},
journal={arXiv preprint arXiv:2405.15364},
year={2024}
}
```