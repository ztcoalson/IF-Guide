## IF-Guide: Influence Function-Guided Detoxification of LLMs

This repository contains the code for IF-Guide, the LLM detoxification technique introduced in our paper:

- [IF-Guide: Influence Function-Guided Detoxification of LLMs](https://arxiv.org/abs/2506.01790)
- **[Zachary Coalson](https://zachcoalson.com)**, Juhan Bae, Nicholas Carlini, [Sanghyun Hong](https://sanghyun-hong.com/)

&nbsp;

----

### TL;DR

You can use our method to detoxify LLMs by identifying harmful training examples and then suppressing them during pre-training or fine-tuning!

&nbsp;

### Abstract

We study how training data contributes to the emergence of toxic behaviors in large-language models. Most prior work on reducing model toxicity adopts reactive approaches, such as fine-tuning pre-trained (and potentially toxic) models to align them with human values. In contrast, we propose a *proactive* approach-IF-Guide-which leverages influence functions to identify harmful tokens within any training data and suppress their impact during training. To this end, we first show that standard influence functions are ineffective at discovering harmful training records. We then present a novel adaptation that measures token-level attributions from training data to model toxicity, along with techniques for selecting toxic training documents and a learning objective that can be integrated into both pre-training and fine-tuning. Moreover, IF-Guide does not rely on human-preference data, which is typically required by existing alignment methods. In evaluation, we demonstrate that IF-Guide substantially reduces both explicit and implicit toxicity-by up to 10× compared to uncensored models, and up to 3× compared to baseline alignment methods, e.g., DPO and RAD-across both pre-training and fine-tuning scenarios. IF-Guide is computationally efficient: a billion-parameter model is not necessary for computing influence scores; a million-parameter model-with 7.5× fewer parameters-can effectively serve as a proxy for identifying harmful data.

&nbsp;

----

## Installation

First, initialize the [Kronfluence](https://github.com/pomonam/kronfluence) package used for running influence functions (we provide a custom implementation that supports *differential influence*):

```bash
git submodule update --init --recursive
```

Create the conda environment (you can use any environment with `python>=3.10`) and install the necessary packages:

```bash
conda create -n IF-Guide python=3.10
conda activate IF-Guide
pip install -r requirements.txt
```

**Note:** We use the package [Kronfluence](https://github.com/pomonam/kronfluence) to compute influence scores with EK-FAC. We create a [custom implementation](https://github.com/ztcoalson/kronfluence) that supports the *differential influence* technique introduced in our paper (page 4, Eq. 6). Credit to all other components of the package goes to the original creators. Thank you!

Next, navigate to the working directory:

```bash
cd src
```

&nbsp;

----

## Model Training/Fine-Tuning

To train a model, run:

```bash
./scripts/train.sh
```

This calls `train.py`, which accepts the following key arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`             | Name of the model to train. Must be registered in `utils/registry.yaml` with a corresponding tokenizer (see existing models for examples) |
| `--save_id`                | Descriptor tag used for output directory naming.        |
| `--toxic_token_mask_path`  | Path to a token mask (generated via IF-Guide). Use `None` for standard training. |
| `--toxic_lambda`           | The strength of the penalty term used by our training objective. |


**Note:** All other arguments can be left at default values to reproduce our experimental setup. This applies to the following sections.

To fine-tune an existing model, run:

```bash
./scripts/finetune.sh
```

This runs `finetune.py`, which uses the following additional arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--checkpoint_dir` | Path to a saved model. If using a pre-trained model, set it to `None`. |
| `--max_steps`      | Maximum steps to fine-tune for.        |

&nbsp;

----

## Running IF-Guide

IF-Guide is composed of four steps: (1) computing the inverse Hessian approximation with [EK-FAC](https://arxiv.org/abs/2308.03296), (2) computing the token-wise *differential influence* scores over the toxic and non-toxic query data (page 4, Eq. 8), (3) selecting influential toxic tokens to suppress during training (page 23, Alg. 1), and (4) suppressing the toxic tokens with our penalty-based training objective (page 5, Eq. 9).

&nbsp;

### 1. Fit the Inverse Hessian Approximation Factors

Run:

```bash
./scripts/fit_factors.sh
```

This calls `fit_factors.py` and takes the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to fit factors on. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--train_indices_path` | Path to the train indices used to train the model (not necessary if using the entire dataset). Must match the exact indices and be in the same order. We provide the indices of our one-billion-token [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) subset and set their path as the default.|
| `--output_dir`           | Path to save the Hessian approximation data. |

&nbsp;

### 2. Compute Token-Wise Scores

Run:

```bash
./scripts/compute_scores.sh
```

This runs `compute_scores.py` with the following key arguments (in addition to most of the arguments used to compute factors):

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to compute scores for. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--save_id`            | Tag appended to the end of the save directory for custom naming. |
| `--save_dir`           | Directory to save scores in (within the original factors directory).
| `--factors_path`       | Path to the directory containing the (inverse) Hessian factors fit in the previous step. |
| `--query_dataset`      | The query dataset for constructing the query gradient. Currently, the only option is `RTP`. | 
| `--toxic_query_indices_path` | Path to the indices from the query dataset pertaining to *toxic* demonstrations. We provide our toxic subset from RTP in `../data/RTP/query_indices/toxic_indices.npy`. | 
| `--nontoxic_query_indices_path` | Path to indices for *non-toxic* queries. We provide our non-toxic subset from RTP in `../data/RTP/query_indices/nontoxic_indices.npy`. |

&nbsp;

### 3. Select Influential Toxic Tokens

Run:

```bash
./scripts/build_toxic_token_mask.sh
```

This runs `build_toxic_token_mask.py`. It takes the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to build the mask for. |
| `--scores_path`     | Path to the scores computed in the prior step. |
| `--window`            | The context window length. |
| `--toxicity_threshold`           | The threshold for determining toxic tokens (as a percentile, e.g., 0.99).
| `--max_tokens`       | The maximum number of toxic tokens to select. |
| `--query_dataset`      | The query dataset for constructing the query gradient. Currently, the only option is `RTP`. | 
| `--inspection_idx` | We automatically print out the suppressed tokens for a single training example in red. This argument specifies which example to print based on its ranking (e.g., 0 is the highest-ranked training example).  | 

&nbsp;

### 4. Suppressing Toxic Tokens During Training/Fine-Tuning

After computing the toxic tokens mask for a particular model, you can specify the `--toxic_token_mask_path` and `--toxic_lambda` arguments in `./scripts/train.sh` (and `./scripts/finetune.sh`) to train/fine-tune models with IF-Guide.

&nbsp;

----

## Evaluation

We provide code for evaluating explicit toxicity (via [Detoxify](https://github.com/unitaryai/detoxify/tree/master)), implicit toxicity (via [ToxiGen-RoBERTa](https://huggingface.co/tomh/toxigen_roberta)), and fluency (measured on [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai) and [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)).

&nbsp;

### Explicit Toxicity Evaluation

Run:

```bash
./scripts/run_toxicity_eval.sh
```

This runs `run_toxicity_eval.py`, which has the following main arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to evaluate. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--dataset`            | Dataset to evaluate on. Either `RTP`, `AttaQ`, or `BOLD`. |
| `--save_dir`           | Directory to save the results in.
| `--decoding_defense`       | Decoding-time defense to apply. `none` or `rad`. Does not apply to our OpenWebText evaluation. |
| `--save_outputs`      | Whether to save the model's outputs. | 

&nbsp;

### Implicit Toxicity Evaluation

Run: 

```bash
./scripts/run_implicit_toxicity_eval.sh
```

It requires the following arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--outputs_file_path`         | The path to an output file from an explicit toxicity run. We re-evaluate the existing outputs to save time. This should be an `output.json` file generated during explicit evaluation. |
| `--dataset`            | Dataset the outputs are from. Determines how the final outputs are formatted. |

&nbsp;

### Fluency Evaluation

Run:

```bash
./scripts/run_fluency_eval.sh
```

It has the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to evaluate. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--dataset`            | Dataset to evaluate on. Either `RTP`, `AttaQ`, or `BOLD`. |
| `--save_dir`           | Directory to save the results in.
| `--decoding_defense`       | Decoding-time defense to apply. `none` or `rad`. Does not apply to our OpenWebText evaluation. |

&nbsp;

### Testing with [RAD](https://github.com/r-three/RAD)

We find that our method is composable with the decoding-time defense [*Reward Augmented Decoding (RAD)* [EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.721/). To run IF-Guide with RAD (or test RAD independently), first download the reward model ([provided by the authors of the original work](https://github.com/r-three/RAD)) and place it in the expected directory:

```bash
cd utils/rad/reward_modeling
gdown https://storage.googleapis.com/rad_release/saved_models.zip
unzip saved_models.zip && rm saved_models.zip && rm -rf saved_models/gpt2_sentiment
```

Credit for the RAD implementation we use belongs entirely to the authors of the original work. Thank you!

&nbsp;

----

## Cite Our Work

Please cite our work if you find this source code helpful.

```
@article{coalson2025ifguide,
  title={IF-Guide: Influence Function-Guided Detoxification of LLMs},
  author={Coalson, Zachary and Bae, Juhan and Carlini, Nicholas and Hong, Sanghyun},
  journal={arXiv preprint arXiv:2506.01790},
  year={2025}
}
```

&nbsp;

----

&nbsp;

Please contact Zachary Coalson (coalsonz@oregonstate.edu) for any questions and recommendations.
