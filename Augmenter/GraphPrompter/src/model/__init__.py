from src.model.graph_llm import GraphLLM
from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.gnn import GCN
from src.model.gnn import GAT
from src.model.llama_adapter import LlamaAdapter
from src.model.t5 import T5

load_model = {
    't5': T5,
    'graph_llm': GraphLLM,
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'gcn': GCN,
    'gat': GAT,
    'llama_adapter': LlamaAdapter,
}


llama_model_path = {
    # '7b': '/gpfsnyu/scratch/ny2208/jch/graphprompter/graphprompter/Llama-2-7b-hf',
    '7b': '/gpfsnyu/scratch/ny2208/jch/graphprompter/graphprompter_recompose/Llama-2-7b-hf',
    '13b': '[Your LLM PATH]',
    '7b_chat': '[Your LLM PATH]',
    '13b_chat': '[Your LLM PATH]',
}
