<div align="center">

<h1>
WiseAD: Knowledge Augmented End-to-End Autonomous Driving with Vision-Language Model
</h1>

<p align="center">
<a href=https://arxiv.org/abs/2412.09951><img src="https://img.shields.io/badge/ArXiv-2412.19505-%23840707.svg" alt="ArXiv"></a>
</p>

Songyan Zhang<sup>1*</sup>, Wenhui Huang<sup>1*</sup>, Zihui Gao<sup>2</sup>, Hao Chen<sup>2</sup>, Lv Chen<sup>1†</sup>

Nanyang Technology University<sup>1</sup>, Zhejiang University<sup>2</sup>

*Equal Contributions, †Corresponding Author

<image src="./assets/framework.png"/><br>
An overview of the framework of our WiseAd.
</div>

## ✨Capabilities

<image src="./assets/WiseAD.png"/>

An overview of the capability of our proposed WiseAD, a specialized vision-language model for end-to-end autonomous driving with extensive
fundamental driving knowledge. Given a clip of the video sequence, our WiseAD is capable of answering various driving-related questions
and performing knowledge-augmented trajectory planning according to the target waypoints.

## 🦙 Data & Model Zoo
Our WiseAD is built on the [MobileVLM V2 1.7B](https://huggingface.co/mtgv/MobileVLM_V2-1.7B) and finetuned on a mixture of datasets including [LingoQA](https://github.com/wayveai/LingoQA), [DRAMA](https://usa.honda-ri.com/drama), and [Carla](https://github.com/opendilab/LMDrive) datasets, which can be downloaded via the related sites.<br>
Our WiseAD is now available at [huggingface](https://huggingface.co/wyddmw/WiseAD). Enjoy playing with it!


## 🛠️ Install

1. Clone this repository and navigate to MobileVLM folder
   ```bash
   git clone https://github.com/wyddmw/WiseAD.git
   cd WiseAD
   ```

2. Install Package
    ```Shell
    conda create -n wisead python=3.10 -y
    conda activate wisead
    pip install --upgrade pip
    pip install torch==2.0.1
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start

#### Example of answering driving-related questions.

```python
python run_infr.py
```

## 🔨 TODO LIST

- [✓] Release hugging face model and inference demo.
- [ ] Carla closed-loop evaluation (coming soon).
- [ ] Training data and code (coming soon).

## Reference

We appeciate the awesome open-source projects of [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM.git) and [LMDrive](https://github.com/opendilab/LMDrive).

## ✏️ Citation

If you find WiseAD is useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@article{zhang2024wisead,
  title={WiseAD: Knowledge Augmented End-to-End Autonomous Driving with Vision-Language Model},
  author={Zhang, Songyan and Huang, Wenhui and Gao, Zihui and Chen, Hao and Lv, Chen},
  journal={arXiv preprint arXiv:2412.09951},
  year={2024}
}
```
