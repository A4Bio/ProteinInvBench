# OpenCPD: Open-source Toolbox for Computaional Protein Design

<p align="left">
<a href="https://arxiv.org/abs/2209.12643" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2209.12643-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/A4Bio/OpenCPD/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<a href="https://github.com/A4Bio/OpenCPD/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/A4Bio/OpenCPD?color=%23FF9600" /></a>
</p>

This repository is an open-source project for computaional protein design benchmarks, which is on updating!

## News and Updates

The project is on updating, please wait for the pre-release version.

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git git@github.com:A4Bio/OpenCPD.git
cd OpenCPD
conda env create -f environment.yml
conda activate OpenCPD
python setup.py develop
```

<details close>
<summary>Dependencies</summary>

* argparse
* fvcore
* numpy
* scikit-learn
* torch
* timm
* torch-geometric
* tqdm
</details>

<!-- Please refer to [install.md](docs/en/install.md) for more detailed instructions. -->

## Getting Started

### Obtaining Dataset

```bash
bash {method}_GetDataset.sh
```
1. CATH
    ```
    cd data/cath
    sh download_cath.sh
    ```

2. AF2 DB
    [Original Dataset](https://alphafold.ebi.ac.uk/.) and [Processed Dataset](https://drive.google.com/drive/folders/1TeojgosleXo3j4sF41vvOjCbOthPQfKm?usp=sharing) can be downloaded here.

### Training
```bash
python main.py --method {method} 
```

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

SimVPv2 is an open-source project for video prediction methods created by researchers in **CAIRI AI Lab** for [A4Bio](https://github.com/A4Bio) projects. We encourage researchers interested in video and weather prediction to contribute to OpenCPD!

## Citation

If you are interested in our repository, please cite the following project:

```
@article{2023opencpd,
  title={OpenCPD: Open-source Toolbox for Computaional Protein Design},
  author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Li, Stan Z},
  journal={arXiv preprint arXiv:2211.12509},
  year={2022}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
