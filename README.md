# PPCU_IFOM_YeastTracker

This aim of this library is to provide ML-based cell tracking solutions for Yeast cells, using a novel time-symmetric tracking approach. It is based on a collaborative research project between [PPCU](https://itk.ppke.hu/en) and [IFOM](https://www.ifom.eu/en/)

Paper publication is still in progress, for the preprint version please check out: 
https://arxiv.org/abs/2308.03887

While publication is in progress, please cite our work as:
'
@article{szabo2023enhancing,
  title={Enhancing Cell Tracking with a Time-Symmetric Deep Learning Approach},
  author={Szab{\'o}, Gergely and Bonaiuti, Paolo and Ciliberto, Andrea and Horv{\'a}th, Andr{\'a}s},
  journal={arXiv preprint arXiv:2308.03887},
  year={2023}
}
'

## Installation Requirements

- Numpy: pip install numpy
- CV2: pip install opencv-python
- Matplotlib: pip install matplotlib
- Skimage: pip install scikit-image
- Scipy: pip install scipy
- PyTorch: pip install torch
- Detectron2: pip install "git+https://github.com/facebookresearch/detectron2.git"
- SMP: pip install segmentation-models-pytorch
- MRC: pip install mrc==0.1.5

## Resources

Models available at:
https://users.itk.ppke.hu/~szage11/IFOM%20tracking/TrainedModels/

Sample data available at:
https://users.itk.ppke.hu/~szage11/IFOM%20tracking/SampleData/

## Getting Started

Example full run:
[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-Open%20Notebook-blue?logo=google-colab)](https://colab.research.google.com/drive/1RezYwQdPQ-eFsBE7oWcIbqv4ylTywhs2?usp=drive_link)