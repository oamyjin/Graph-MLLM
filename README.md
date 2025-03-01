



<p align="center">
  <a href="#Graph-MLLM">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="">Examples</a> •
  <a href="">Docs</a>
</p>

# Graph-MLLM

Official code for ***Graph**-**MLLM**: Harnessing Multimodal Large Language Models for Multimodal **Graph** Learning*. Graph-MLLM is a comprehensive benchmark for multimodal graph learning that extends state-of-the-art graph methods into the multimodal domain using MLLMs. Fusing multimodal data with graph-structured data shows great promise for numerous real-world applications—such as social networks, healthcare, and recommendation systems—when nodes contain both textual and visual attributes.


## Overview of Graph-MLLM

Graph-MLLM provides a fair and comprehensive platform to evaluate existing graph learning methods and facilitate future research on multimodal graph learning.

![graphmllm](https://github.com/oamyjin/Graph-MLLM/blob/main/docs/graphmllm.pdf)


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
You can find dataset from ![huggingface](https://huggingface.co/datasets/oamyjin/Graph-MLLM/tree/main/enhancer-dataset)


**Run Evaluation**
You can use the command `./Enhancer/run_all.sh` to run Enhancer on all datastets.
