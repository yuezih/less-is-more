<div>
  <h2 align="center">
    Less is More
  </h2>
</div>

<p align="center">
    <a >
       <img alt="Issues" src="https://img.shields.io/github/issues/yuezih/less-is-more?color=blueviolet" />
  	</a>
    <a >
       <img alt="Forks" src="https://img.shields.io/github/forks/yuezih/less-is-more?color=orange" />
  	</a>
    <a >
       <img alt="Stars" src="https://img.shields.io/github/stars/yuezih/less-is-more?color=ff69b4" />
  	</a>
    <a >
      <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2402.14545-b31b1b?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2402.14545">
  	</a>
    <br />
</p>


[Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective](https://arxiv.org/abs/2402.14545)

---

<!-- > To be contunued... -->

## Selective EOS Supervision

### Training

Following the instruction of [LLaVA](https://github.com/haotian-liu/LLaVA) to prepare the environment, data (`LLaVA-Instruction-150K`) and pretraining models (e.g., `LLaVA-1.5-7b`). 

Train the model with Selective EOS Supervision. The default configuration is set to train the `llava-1.5-7b` model with `Detail23k` for one epoch.

```bash
cd LLaVA
bash scripts/v1_5/selective_eos_finetune.sh
```
The main modifications to the original LLaVA code for Selective EOS Supervision is detailed in [./docs/selective-eos-supervision.md](./docs/selective-eos-supervision.md).

### Checkpoint

Our models finetuned with Selective EOS Supervision:

Basic Model | Finetuning Data | Checkpoint
 :- | :- | :-
`llava-1.5-7b` | `Detail23k` | [llava-v1.5-7b-selective-23k](https://huggingface.co/yuezih/llava-v1.5-7b-selective-23k)
`llava-1.5-7b` | `LLaVA-Instruction-150K` | [llava-v1.5-7b-selective-150k](https://huggingface.co/yuezih/llava-v1.5-7b-selective-150k)

## CHAIR Evaluation

### Data

The test set used in our paper for CHAIR evaluation is provided in [./CHAIR-eval/data/chair-500.jsonl](./CHAIR-eval/data/chair-500.jsonl). The data is randomly sampled from the MSCOCO validation set with a random seed of 0.

For test set images, we provide a [python script](./CHAIR-eval/prepare_data.py) to collect images from the original MSCOCO images with softlinks. Please specify the path of your own MSCOCO image path. The script will create a folder `./CHAIR-eval/data/chair-500` for the CHAIR images.

```bash
python ./CHAIR-eval/prepare_data.py
```
The script also downloads the annotation files of MSCOCO detection, which will be used for CHAIR evaluation.

### Evaluation

We provide a [script](./CHAIR-eval/eval.sh) for CHAIR inference and evaluation.  
Set your model in the following script and then run it:

```bash
bash ./CHAIR-eval/eval.sh
```
The first-time evaluation can be slow because of the ground-truth object set construction. Subsequent evaluations will be faster with the cache.


## Citation

If you find this repo helpful, please consider citing our paper:

```bibtex
@misc{yue2024less,
      title={Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective}, 
      author={Zihao Yue and Liang Zhang and Qin Jin},
      year={2024},
      eprint={2402.14545},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

This repo is built on [LLaVA](https://github.com/haotian-liu/LLaVA) (models) and [OPERA](https://github.com/shikiw/OPERA) (CHAIR evaluation). Many thanks for their efforts. The use of our code should also follow the original licenses.