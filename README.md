## Zero-Shot Cross-Lingual Text Classification with Robust Training

Code for our EACL-2021 paper ["Improving Zero-Shot Cross-Lingual Transfer Learning via Robust Training"](https://arxiv.org/abs/2104.08645).
Most of our code is based on [XTREME](https://github.com/google-research/xtreme).

The current codebase only supports `randomized smoothing with data augmentation` because it's the most effective method that has consistent improvements over different tasks in our paper. We will consider adding other methods and baselines in this repository.


### Setup 

- Python 3.7+
```
bash install_tools.sh
```
If you encounter any issues when installing the environment, please refer to [XTREME](https://github.com/google-research/xtreme).

### Data

Download [data](https://drive.google.com/file/d/184VriHkpfffWPSTcZf6wUm8ezz7RIUsX/view?usp=sharing) and unzip it. It includes the original PAWS-X and XNLI dataset as well as the testing set for the *generalized* setting.

Run the following commands to generate the augmented data for randomized smoothing.
```
python perturb.py --task pawsx --input_dir data_generalized --output_dir data_generalized_augment --num 10
python perturb.py --task xnli --input_dir data_generalized --output_dir data_generalized_augment --num 3
```

### Training

```
./scripts/train_pawsx.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir]
./scripts/train_xnli.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir]
```

### Evaluataion

For standard setting
```
./scripts/eval_pawsx.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir] [model_dir]/checkpoint-best/
./scripts/eval_xnli.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir] [model_dir]/checkpoint-best/
```

For generalized setting
```
./scripts/eval_generalized_pawsx.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir] [model_dir]/checkpoint-best/
./scripts/eval_generalized_xnli.sh bert-base-multilingual-cased [gpu_id] data_generalized_augment [output_dir] [model_dir]/checkpoint-best/
```


### Citation

If you find that the code is useful in your research, please consider citing our paper and the XTREME paper.

    @inproceedings{Huang2021robust-xlt,
        author    = {Kuan-Hao Huang and
                     Wasi Uddin Ahmad and 
                     Nanyun Peng and
                     Kai-Wei Chang},
        title     = {Improving Zero-Shot Cross-Lingual Transfer Learning via Robust Training},
        booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
        year      = {2021},
    }
    
    @inproceedings{Hu20xtreme,
        author    = {Junjie Hu and
                     Sebastian Ruder and
                     Aditya Siddhant and
                     Graham Neubig and
                     Orhan Firat and
                     Melvin Johnson},
        title     = {{XTREME:} {A} Massively Multilingual Multi-task Benchmark for Evaluating
                     Cross-lingual Generalisation},
        booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
        year      = {2020},
      }
      
