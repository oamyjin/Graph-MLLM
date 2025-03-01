<div align="center">
<img src="https://github.com/oamyjin/Graph-MLLM/blob/main/docs/title.png" border="0" width=400px/>
</div>

------
<p align="center">
  <a href="#Graph-MLLM">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="">Examples</a> •
  <a href="">Docs</a>
</p>



## Overview

Official code for ***Graph**-**MLLM**: Harnessing Multimodal Large Language Models for Multimodal **Graph** Learning*. 

Graph-MLLM is a comprehensive benchmark for multimodal graph learning that extends state-of-the-art graph methods into the multimodal domain using MLLMs. Fusing multimodal data with graph-structured data shows great promise for numerous real-world applications—such as social networks, healthcare, and recommendation systems—when nodes contain both textual and visual attributes.

![graphmllm](https://github.com/oamyjin/Graph-MLLM/blob/main/docs/graphmllm.png)


## Installation

``` bash
git clone https://github.com/oamyjin/Graph-MLLM.git
```

**Installation for Enhancer:**
``` bash
cd Enhancer
pip install -r requirement.txt
```

## 🚀Quick Start
Here we shows a quick start to run Enhancer evalution.

**Update Dataset**
You can find dataset from [huggingface](https://huggingface.co/datasets/oamyjin/Graph-MLLM/tree/main/enhancer-dataset)


**Run Evaluation**
You can use the command `./Enhancer/run_all.sh` to run Enhancer on all datastets.

## Step 1: Environment Preparation 

```shell
# create a new environment
conda create -n llaga python=3.10
conda activate llaga

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt

# install flash-attn
pip install flash-attn --no-build-isolation

# install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

```
## Step 2:  Data Preparation
Download our datasets from [huggingface](https://huggingface.co/datasets/oamyjin/Graph-MLLM/tree/main/enhancer-dataset). And move the processed data to `./dataset`
```
.
├── configs
├── dataset
│   ├── Arts
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   ├── CD
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   └── pubmed
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
│   ├── cora
│   │   ├── clip_feat.pt
│   │   ├── clipimage_feat.pt
│   │   ├── clipnonstruc_feat.pt
│   │   ├── clipstruc_feat.pt
│   │   ├── cliptext_feat.pt
│   │   ├── labels-w-missing.pt
│   │   ├── nc_edges-nodeid.pt
│   │   ├── node_mapping.pt
│   │   ├── split.pt
├── main.py
├── models.py
├── nc_dataset.py
├── requirements.txt
├── run_all.sh
```
