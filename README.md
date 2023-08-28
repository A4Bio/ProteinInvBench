```

```

# ProteinInvBench: Benchmarking Protein Design on Diverse Tasks, Models, and Metrics
Model zoom: https://zenodo.org/record/8031783
<p align="left">
<!-- <a href="https://arxiv.org/abs/2211.12509" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2211.12509-b31b1b.svg?style=flat" /></a> -->
<a href="https://github.com/A4Bio/OpenCPD/blob/release/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a> -->
<a href="https://github.com/A4Bio/OpenCPD/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/A4Bio/OpenCPD?color=%23FF9600" /></a>
<a href="https://github.com/A4Bio/OpenCPD/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/A4Bio/OpenCPD/" alt="arXiv">
    <img src="https://img.shields.io/github/stars/A4Bio/OpenCPD" /></a>
</p>

[📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🆕News](docs/en/changelog.md)

This repository is an open-source project for benchmarking structure-based protein design methods, which provides a variety of collated datasets, reprouduced methods, novel evaluation metrics, and fine-tuned models that are all integrated into one unified framework. It also contains the implementation code for the paper:

**ProteinInvBench: Benchmarking Protein Design on Diverse Tasks, Models, and Metrics**

[Zhangyang Gao](https://scholar.google.com/citations?user=4SclT-QAAAAJ&hl=en), [Cheng Tan](https://chengtan9907.github.io/), [Yijie Zhang](https://scholar.google.com/citations?user=Q9Gby5wAAAAJ&hl=en), [Xingran Chen](https://www.chenxingran.com/), [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl).

## Introduction

[ProteinInvBench]() is the first comprehensive benchmark for protein design. The main contributions of our paper could be listed as four points below:

* **Tasks:** We extend recent impressive models from single-chain protein design to the scenarios of multi-chain and de-novoprotein design.
* **Models:** We integrate recent impressive models into a unified framework for efficiently reproducing and extending them to custom tasks.
* **Metrics:** We incorporate new metrics such as confidence, sc-TM, and diversity for protein design, and integrate metrics including recovery, robustness, and efficiency to formulate a comprehensive evaluation system.
* **Benchmark:** We establish the first comprehensive benchmark of protein design, providing insights into the strengths and weaknesses of different methods.

<p align="center">
    <img width="75%" src=https://s1.ax1x.com/2023/06/14/pCnlp9K.jpg> <br>
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- <p align="center">
    <img width="75%" src=https://github.com/A4Bio/OpenCPD/blob/release/assets/CATH.png> <br>
</p> -->

## Overview

<details open>
<summary>Major Features</summary>

- **Unified Code Framework**
  ProteinInvBench integrates the protein design system pipeline into a unified framework. From data preprocessing to model training, evaluating and result recording, all the methods collected in this paper could be conducted in the same way, which simplifies further analysis to the problem. In detail, ProteinInvBench decomposes computational protein design algorithms into `methods` (training and prediction), `models` (network architectures), and `modules. `Users can develop their own algorithms with flexible training strategies and networks for different protein design tasks.
- **Comprehensive Model Implementation**
  ProteinInvBench collects a wide range of recent impressive models together with the datasets and reproduces all the methods in each of the datasets with restricted manners.
- **Standard Benchmarks**
  ProteinInvBench supports standard benchmarks of computational protein design algorithms with various evaluation metrics, including novel ones such as confidence, diversity, and sc-TM. The wide range of evaluataion metrics helps to have a more comprehensive understanding of different protein design algorithms.

</details>

<details open>
<summary>Code Structures</summary>

- `run/` contains the experiment runner and dataset configurations.
- `configs/` contains the model configurations
- `opencpd/core` contains core training plugins and metrics.
- `opencpd/datasets` contains datasets and dataloaders.
- `opencpd/methods/` contains collected models for various protein design methods.
- `opencpd/models/` contains the main network architectures of various protein design methods.
- `opencpd/modules/` contains network modules and layers.
- `opencpd/utils/` contains some details in each model.
- `tools/` contains the executable python files and script files to prepare the dateset and save checkpoints to the model.

</details>

<details open>
<summary>Demo Results</summary>
 The result of methods collected on CATH dataset is listed as following:
<p align="center">
    <img width="100%" src=https://s1.ax1x.com/2023/06/19/pC1r6ts.png> <br>
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

</details>

## News and Updates

[2023-06-16] `ProteinInvBench` v0.1.0 is released.

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:

```shell
git clone https://github.com/A4Bio/OpenCPD.git
cd opencpd
conda env create -f environment.yml
conda activate opencpd
python setup.py develop
```

## Getting Started

**Obtaining Dataset**

The processed datasets could be found in the [releases](https://www.idrive.com/idrive/sh/sh?k=p9b2y3l6i5). 
*To note that, due to the large file size, ProteinMPNN dataset was uploaded in a separate file named mpnn.tar.gz, others could be found in data.tar.gz*

**Model Training**

```shell
python main.py --method {method} 
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Supported Models, Datasets, and Evaluation Metrics

We support various protein design methods and will provide benchmarks on various protein datasets. We are working on adding new methods and collecting experiment results.

<!-- *The detailed introduction could be found in* [dataset.md]() -->

* Protein Design Methods.

  <details open>
    <summary>Currently supported methods</summary>

  The models and their corresponding results and checkpoints can be downloaded [here](https://zenodo.org/record/8031783)

  - [X] [GraphTrans](https://proceedings.neurips.cc/paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html) (NeurIPS'2021)
  - [X] [StructGNN](https://arxiv.org/abs/2011.12117) (NeurIPS'2020)
  - [X] [GVP](https://arxiv.org/pdf/2303.11783.pdf)
  - [X] [GCA](https://arxiv.org/abs/2204.10673) (ICASSP'2023)
  - [X] [AlphaDesign](https://arxiv.org/abs/2202.01079)
  - [X] [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) (Scienece)
  - [X] [PiFold](https://arxiv.org/abs/2209.12643) (ICLR'2023)
  - [X] [KWDesign](https://arxiv.org/abs/2305.15151)

  </details>

  <details open>
    <summary>Currently supported datasets</summary>

  To download the processed datasets, please click [here](https://www.idrive.com/idrive/sh/sh?k=p9b2y3l6i5)
  The details of the datasets could be found in [dataset.md]()

  - [X] CATH
  - [X] PDB
  - [X] CASP15

  </details>
  <details open>
    <summary>Currently supported evaluation metrics</summary>

  - [X] Recovery
  - [X] Confidence
  - [X] Diversity
  - [X] sc-TM
  - [X] Robustness
  - [X] Efficiency

  </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

ProteinInvBench is an open-source project for structure-based protein design methods created by researchers in **CAIRI AI Lab**. We encourage researchers interested in protein design and other related fields to contribute to this project!

## Citation

```
@article{gao2023knowledge,
  title={Knowledge-Design: Pushing the Limit of Protein Design via Knowledge Refinement},
  author={Gao, Zhangyang and Tan, Cheng and Li, Stan Z},
  journal={arXiv preprint arXiv:2305.15151},
  year={2023}
}
```


## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with `ProteinInvBench`, please open a [GitHub issue](https://github.com/A4Bio/OpenCPD/issues) and [pull request](https://github.com/A4Bio/OpenCPD/pulls) with the tag "new features", "help wanted", or "enhancement". Feel free to contact us through email if you have any questions.

- Zhangyang Gao (gaozhangyang@westlake.edu.cn), Westlake University & Zhejiang University
- Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University
- Yijie Zhang (yj.zhang@mail.mcgill.ca), McGill University
- Xingran Chen(chenxran@umich.edu), University of Michigan

<p align="right">(<a href="#top">back to top</a>)</p>
