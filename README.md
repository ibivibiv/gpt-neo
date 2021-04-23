# GPT Neo

🎉 1T or bust my dudes 🎉

An implementation of model & data parallel [GPT2](https://openai.com/blog/better-language-models/) & [GPT3](https://arxiv.org/abs/2005.14165)-like models, with the ability to scale up to full GPT3 sizes (and possibly more!), using the [mesh-tensorflow](https://github.com/tensorflow/mesh) library.

Training and inference supported on both TPUs and GPUs.

Also included are alternative model architectures and linear attention implementations that should enable scaling up to even larger model sizes & context lengths, including:

* Local attention
* [Linear attention](https://arxiv.org/abs/1812.01243)
* [Mixture of Experts](https://arxiv.org/abs/1701.06538)
* [Axial Positional embedding](https://arxiv.org/abs/1912.12180)
* Masked Language Modelling

Pretrained models will be released as they are finished training.


# Setup for NVidia GPUS on Ubuntu

There are several steps that need to be followed to untangle setting up a proper mesh configuration and memory limits for using the TPU  models expressed in the majority of the documentation.  The examples are mostly geared towards using Google Cloud TPU modules.  I successfully have gotten a decent model training on NVidia GPUs on a server.  It involves mostly installing the correct Python, Cuda, and cudnn versions.  After that you must set your config for the model training to have the right mesh config and bring some of the values into tolerance for the available memory in your setup.  I'll document what I did for this version here:

<i>STARTING WITH A MINIMAL UBUNTU 18.04</i>

## Python 
I took python from 3.6.x to 3.8 first by doing the following:

* sudo apt-get install software-properties-common
* sudo add-apt-repository ppa:deadsnakes/ppa
* sudo apt-get update
* sudo apt-get install python3.8
* sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
* sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
* sudo update-alternatives --config python3
* at this point you select "2" or you can just use default if it has change 3.8 to the default
* apt-get install python3-pip
* pip install --upgrade pip
* upgrading pip is important because things like tensorflow-mesh aren't there in early versions of pip

## Cuda
I downloaded this script to install Cuda 11 but its also here in the repo to use
* sudo apt-get install git
* sudo git clone https://github.com/DataCrunch-Scripts/Install-CUDA.git ~/Install-CUDA
* select version 11
* sudo chmod +x ~/Install-CUDA/installer.sh
* sudo ~/Install-CUDA/installer.sh
* reboot your system
* check that your system is seeing the GPU's by executing "nvidia-smi"

## Cudnn
* wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
* chmod a+x libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
* dpkg -i libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb


## Tensorflow
* python3 -m pip install tensorflow-gpu


The I ran the following quick python script to ensure I had Python, Cuda, Cudnn, and Tensorflow all working:

<i>
from tensorflow.python.client import device_lib <br/>
<br/>
def get_available_gpus():<br/>
    local_device_protos = device_lib.list_local_devices()<br/>
    return [x.name for x in local_device_protos if x.device_type == 'GPU']<br/>
<br/>
print(get_available_gpus())<br/>
</i>

## GPTNeo
* git clone https://github.com/EleutherAI/GPTNeo
* cd GPTNeo
* pip3 install -r requirements.txt



## Command

Then I used this command to train:

nohup python3 main.py --model colab_XL --steps_per_checkpoint 500 --gpu_ids device:GPU:0 &




## Tokenizing your Dataset

If you just want to test training, you can skip this step and download some dummy data like so:

```
wget https://storage.googleapis.com/connors-datasets/bundestag/bundestag_0.tfrecords
```

Then copy the data to your bucket, or if using GPUs, a local directory: 

```
gsutil cp bundestag_0.tfrecords gs://<your bucket>/
```

If using your own data to train, you can use the `data/create_tfrecords.py` script to encode your text data into tfrecords.

Your data must either be in the form of lots of normal .txt files (one document per file), or in any format supported by [lm_dataformat](https://github.com/leogao2/lm_dataformat). 

You can run the script without parameters to see help for all options.

In **document mode** Each example in the tfrecords is one (variably sized) document. This is to be used with the `documents_fixed` and `documents_random` sampling modes (For more details see the parameters reference section).
Document mode is the default mode.

The below command will tokenize all files in acceptable formats in *base_dir* using gpt2 tokenizer and save them to *output_dir*
```
python3 create_tfrecords.py --mode documents --input_dir <base> --name <name> --output_dir <output> --use_gpt2_tokenizer --minimum_size <min> 
```

- `input_dir`: Defines the folder where your data is located. The script will encode all files present in this folder.
- `name`: Name of output files will be `name_i.tfrecords` where i is the number of the file.
- `output_dir`: Where to save the tfrecords to
- `use_gpt2_tokenizer`: Whether to use the pretrained HuggingFace GPT2 tokenizer, in which case the separator will be set to [50256].
- `encoder_path`: if not using the pretrained gpt2 tokenizer, use this flag to provide a path to your generated tokenizer json.
- `separator`: Written in list format, the separator token(s) to insert between documents (e.g. "[0]"). Will depend on your encoder.
- `minimum_size`: The minimum size (in tokens) a document must have, otherwise it is discarded. This is what will later determine your `stitch` parameter: `stitch * minimum_size` must always be greater or equal `n_ctx` (For more details see the parameters reference section).

## 4. Using a Dataset in a Model

To use a dataset in a model, you must first register that dataset under `./configs/dataset_configs` folder. First choose a filename with a `.json` extension. That filename will serve as the dataset identification. The config should be filled out the following manner.

If you have a dataset encoded using the pretrained gpt2 tokenizer, you can specify that like so:

```json
{
    "n_vocab": 50257,
    "path": "gs://neo-datasets/openwebtext-documents/openwebtext_*.tfrecords",
    "eval_path": "gs://neo-datasets/openwebtext-documents/openwebtext_*.tfrecords",
    "tokenizer_is_pretrained": true,
    "tokenizer_path": "gpt2"
}
```

or if you've trained a custom tokenizer, like so:

```json
{
    "n_vocab": 32768,
    "path": "./path/to/your/*.tfrecords",
    "eval_path": "./path/to/your/eval/*.tfrecords",
    "tokenizer_path": "./path/to/your/byte-level-bpe.tokenizer.json"
}
```

Finally, in your model config, add the filename that you created above to the `datasets` array.

The `<dataset id>` will be the filename, excluding the `.json`, that you created above

```
"datasets": [[<dataset id>, <stitch>, <datatype>, <weight>]] # datasets key defines at run time how each dataset is processed for training
```

## 5. Choose a model configuration

Once you have your datasets set up, find a suitable config in `/configs`.

Here we use a GPT3-XL sized model as an example, but there are many more in `./configs`, all of which have short summaries in the Available Configs section.

All you need to do is edit the dataset id as described above, and edit `model_path` (where logs and checkpoints will be saved) to point to a cloud bucket you have write access to (or local path, if using GPUs).

```json
{
    "n_head": 32,
    "n_vocab": 50257,
    "embed_dropout": 0.1,
    "lr": 0.0002,
    "lr_decay": "cosine",
    "warmup_steps": 3000,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "opt_name": "adam",
    "weight_decay": 0.1,
    "train_batch_size": 512,
    "attn_dropout": 0.1,
    "train_steps": 286150,
    "eval_steps": 0,
    "predict_steps": 1,
    "res_dropout": 0.1,
    "eval_batch_size": 128,
    "predict_batch_size": 1,
    "iterations": 2500,
    "n_embd": 2048,
    "datasets": [["your_dataset_name", 25, "documents_random", 1.0]],
    "model_path": "gs://neo-models/GPT3_XL",
    "n_ctx": 2048,
    "n_layer": 24,
    "scale_by_depth": true,
    "scale_by_in": false,
    "attention_types" :  [[["global"],24]],
    "mesh_shape": "x:128,y:2",
    "layout": "batch:x,memory_length:y,embd:y",
    "activation_function": "gelu",
    "recompute_grad": true,
    "gradient_clipping": 1.0,
    "tokens_per_mb_per_replica": 2048
}
```


## 6. Run Training

```
python3 main.py --model <your_config_name> --steps_per_checkpoint <n> --gpu_ids <device:GPU:0 device:GPU:1>
```


- `steps_per_checkpoint`: The frequency in steps at which to save checkpoints.
- `--auto_layout` and `--auto_layout_and_mesh_shape` (Optional): Disable training and instead auto generate a memory efficient `layout` (and `mesh_shape`)
- `gpu_ids`: if training using GPUs, omit the `tpu` flag and pass in the ids of your gpus. In the example below, we train on 3 GPUs, specifying their device ids delimited by spaces:


# Available Configs

We have several model sizes available, but some of our configs require large TPUs and will need tweaking to run on smaller machines, or GPUs. Below is a short guide to each model in the configs directory:

TODO

# Extra Features: 

## Training (with Sacred)

[Sacred](https://github.com/IDSIA/sacred) helps track experiments and is much nicer to work with than tensorboard.

To setup:

1. Install Docker and Docker-compose

2. Run `docker-compose up`

To use: 

1. Ensure model_dir doesn't have any metric logs in it (it trips up the metric stuff for tensorboard, which assumes that it's a continuation of the existing run). You can use `gsutil rm -r ...` to delete model dir

2. Run `python3 run_experiment.py --tpu sometpuhere --model someconfig.json` Options are the same as `main.py`. 

3. You can go to http://server_ip_goes_here:8081/ to see the Omniboard overview. If you prefer to see a tensorboard, the script also spins one up and automatically assigns it a port. The script should print out the tensorboard port near the top of the log. 

## Peeking at a Dataset

If you are ever confused by the dataset of a particular config file, you can easily check the minimum and maximum token ids with a single command. This is useful for making sure that the vocabulary size of the model is at least as large as the maximum token id. Tensorflow will not error if you try to gather on a matrix with out of bounds indices, so you need to make sure your vocabulary size is sufficiently large.

```bash
python main --model {config_name} --check_dataset
```

## Masked Language Modeling

In addition to being able to train large GPT's, this repository also allows you to easily do masked language modeling (BERT, RoBERTa). In order to do so, you must follow two additional steps.

1. When tokenizing your dataset, you must reserve a special id for the `[mask]` token.

2. In the configs, you will have to define two additional fields

```python
"mlm_training": true,                           # must be set to true
"mlm_mask_id": <mask id>                        # the mask id that you reserved from above
```

That's all you need to train a model with the MLM objective, good for any type of data that you have encoded properly. If you would like to tweak the other related hyperparameters, please continue reading.

```python
"mlm_cls_token_id": <cls token id>,                # auto append specified CLS token id on the left
"mlm_mask_prob": 0.15,                             # the probability of masking a token, defaults to 15%
"mlm_same_token_prob": 0.10,                       # probability of keeping the token the same, defaults to 10%
"mlm_random_token_prob": 0.10,                     # probability of tokens that are replaced with random tokens, 10% was recommended by the BERT paper
"mlm_mask_ignore_ids": [<cls token>, <sep token>]  # ignore masking other special tokens, if any
```

## Parameter Reference

Pick a valid config from `/configs` and tweak the parameters as needed:

- `n_heads`: The number of attention heads.
- `n_embd`: Size of the hidden layers, must be divisible by `n_heads`.
- `n_vocab`: Vocabulary size.
- `embed_dropout`, `res_dropout`, `attn_dropout`: Dropout probability for word embedding/residuals/attention
- `lr`: Learning rate
- `warmup_steps`: Number of steps before full learning rate is reached (linear ramp from `0` to `lr`).
- `lr_decay`: `cosine` or `linear`.
- `opt_name`: `adam` or `adafactor`.
- `beta1`, `beta2` and `epsilon`: `adam` optimizer params.
- `beta1`, `ada_epsilon1` and `ada_epsilon2`: `adafactor` optimizer params.
- `weight_decay`: Weight decay parameter, if not present no weight decay is used (the weight decay fix for Adam is used) (default: 0.01) (optional).
- `train_batch_size`: Batch size during training.
- `train_steps`: Number of training steps (batches), set to roughly ~1 epoch for now (total number of tokens in your dataset / number of tokens per batch (= `train_batch_size` / `n_ctx`)).
- `eval_steps`: Number of steps to run for each evaluation. Set to `0` for no eval. i.e After every checkpoint, the model is tested for `eval_steps`
- `iterations`: Number of steps queued to the TPU, must be smaller than `steps_per_checkpoint`. (default: 500)
- `datasets`: List of tfrecords datasets to use. Each dataset is a list with the following parameters: `[train glob , eval glob, stitch, sampling_mode, weight]`. So for example for a single dataset (note the double list): `[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]`
    + `dataset_id`: The name of a dataset configuration file in `./configs/dataset_configs`
    + `stitch`: If `sampling_mode` `random_sample` is used, the input pipeline samples this amount of texts into one to sample from. You must select stitch so that `stitch * minimum_document_length >= n_ctx`
    + `sampling_mode`: `chunks` (tfrecords are preprocessed into the correct length and are read sequentially) or `documents_random` (`stitch` amount of documents are concatenated and then a `n_ctx` chunk is randomly subsampled)
    + `weights`: How much relative weight this dataset should have compared to others
- `model`: Which model to train. Currently only `GPT` is supported, and it defaults to this if not present.
- `model_path`: local path, if using GPUs to save model checkpoints and logs.
- `n_ctx`: Size of context window. Default is 2048
- `n_layer`: Number of layers (blocks) in the model.
- `scale_by_depth`: If true, the weight initialization of layers are scaled by their depth as in the GPT2 paper.
- `scale_by_in`: If true, the weight initialization of layers are scaled by their number of inputs as in the GPT2 paper.
- `mesh_shape`: A Mesh is an n-dimensional array of processors with named dimensions used for parallelism in the mesh-tensorflow library. Each Tensor is split evenly across mesh dimensions according to the layout (see below). The 'mesh_shape' is the shape of this array, and must be equal to the number of processors. e.g., for a v3-128 TPU "mesh_shape": “x:16,y:8”.
- `layout`: A Tensor is laid out on its mesh with one slice on each processor. A Tensor "layout", is an injective partial map specifying which dimensions of the tensor are (evenly) split across which dimensions of the mesh. No dimension of a tensor may be split across two dimensions of its mesh and no two dimensions of a tensor may be split across the same dimension of its mesh. The user defines a global set of layout rules in the form of (tensor-dimension-name, mesh-dimension-name) pairs. A dimension of a tensor is split across a dimension of its mesh if there is a matching rule, e.g. (for the above example mesh_shape: "layout":"batch:x,heads:y"
- `activation_function`: `selu` (self normalizing) or `gelu` (used by OA), activation function used in feed-forward passes. (default: gelu)
- `attention_types`: the type of attention for each layer in a list of the following format [[["attention_type"], n_layers]]. e.g. for a 12 layer net [[["global"], 12]] or [[["local"], 10], [["global"], 2]].
    + Choose from: `linear`, `global`, `local` or `none`. We have found a 50/50 mix of `global` and `linear` to work well. `none` allows you to create feed-forward only layers for more efficient [PAR Transformer](https://arxiv.org/abs/2009.04534) models.
- `precision`: `float32` or `bfloat16`.
- `tokens_per_mb_per_replica`: If not None, will split the batch up into smaller microbatches containing `tokens_per_mb_per_replica` tokens to avoid OOMs. Gradients are accumulated locally and reduced once. IMPORTANT: mb refers to *minibatch* not megabyte here. 

**Mixture of Experts**

- `moe_layers`: A list of layer numbers to append a [mixture of experts](https://arxiv.org/abs/1701.06538) layer onto. E.G: `[2,4,6,8,10,12]`.
We have experimentally found a moe layer for every two self-attention layers to work well.
-  `moe_params`: a dictionary of additional kwargs to pass in to the moe layer. E.G
    `{"moe_dropout_rate": 0.0 }`
    
**Experimental features** 

- `axial_pos_emb_`: If true, uses [axial positional embedding](https://arxiv.org/abs/1912.12180. 
- `mlp_glu`: If true, uses a gated linear unit variant of feed forward layers.
- `scalenorm`: If true, uses scalenorm instead of layernorm.
- `rezero`: If true, uses [rezero](https://www.groundai.com/project/rezero-is-all-you-need-fast-convergence-at-large-depth/1) instead of layernorm.
- `num_mem_kv`: adds memory / key values from the [all-attention paper](https://arxiv.org/pdf/1907.01470.pdf). Param is an int with the number of desired mem/key values.
- `macaron`: if true - uses a [macaron transformer](https://arxiv.org/pdf/1906.02762.pdf) for each layer block.

## TODO: 

- [x] finalize documentation
- [ ] update configs
