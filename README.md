![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# TokManGAN
codes for TokManGAN: [Token Manipulation Generative Adversarial Network for Text Generation](https://arxiv.org/pdf/2005.02794.pdf) 

This is a hierarchical sequence generation model, that first decides whether a blank is added or not, 

if a blank is added then fill it, otherwise decides how utilize a given token (use, ignore, etc). 

## Instructions
##### 1. Run TokManGAN in MLE mode
> `python train.py --gan_model tokmangan --mode MLE --dataset coco --unit_size 32` 
##### 2. Run TokManGAN in GAN mode
> `python train.py --gan_model tokmangan --mode GAN --dataset coco --unit_size 32` 
##### 3. Generate samples
* For evaluation
> `python generate_for_eval.py -g tokmangan -t GAN -d coco -s 32`
* For checking details (You can find result files that start with 'details' in <a href="./save/coco_tokmangan">here</a>)
> `python generate_for_details.py -g tokmangan -t GAN -d coco -s 32 --n_generate_per_seed 10 --gen_vd_keep_prob 0.8`

+ You can also produce MaskGAN model using on this project. (specify the option in the script as *--gan_model maskgan* or *-g maskgan*)

## Pre-trained model(MLE mode)
| Model URL                                                                 | Data       | Steps        |
|---------------------------------------------------------------------------|------------|--------------|
| [link](https://drive.google.com/open?id=1Sr7zah3GC9ekLqsgT3qlF1vLQ9KdRkVD)| Image COCO | 80 epochs    |
|  | EMNLP News |    |


## Synthetic data experiment
* You can find text samples synthesized in <a href="./save/coco_tokmangan">here</a>
* Codes for evaluation are placed in <a href="./evaluation.ipynb">here</a>

## Requirements
* Code is tested on TensorFlow version 1.14 for Python 3.6.
* For evaluation you need to download the external project - [GansFallingShort](https://github.com/pclucas14/GansFallingShort)

## References
We have helped a lot in the following projects.
- [MaskGAN](https://github.com/tensorflow/models/tree/master/research/maskgan)
- [Texygen](https://github.com/geek-ai/Texygen)
- [GansFallingShort](https://github.com/pclucas14/GansFallingShort)
