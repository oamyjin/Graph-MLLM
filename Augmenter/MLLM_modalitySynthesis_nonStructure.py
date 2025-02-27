import os
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoProcessor
import torch
from datasets import load_dataset
import numpy as np
import ast
import json
import argparse
import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/js12556/mgllm/Qwen-VL/local_model/Qwen-VL-Chat")
    parser.add_argument("--cvs-file", type=str, default='/scratch/js12556/mgllm/output_image2summary/CD/cd_aug.csv')
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=100000)
    parser.add_argument("--dataset-name", type=str, default="CD")
    parser.add_argument("--use-image", type=int, default=1)
    parser.add_argument("--use-text", type=int, default=0)
    parser.add_argument("--output-file", type=str, default="/scratch/js12556/mgllm/output_image2summary/CD/CD_structure_missing_2.json")
    args = parser.parse_args()

    args.use_image = bool(args.use_image)
    args.use_text = bool(args.use_text)
    
    dataset_name = args.cvs_file.split("/")[-1].split(".")[0]
    checkpoint = args.model_path
    print("dataset_name:", dataset_name)
    print("use_image:", args.use_image, ", use_text:", args.use_text)
    print("local_model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    print("tokenizer done")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
    print("model done")
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
    print("model.generation_config done")
    # processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    # print("processor done")
    model.generation_config.top_p = 0.01 # TODO: What is 0.01???
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("device:", device)

    prompts = [" \nQuestions: Using the title, description, and image summary of the product provided above, create an informative and concise description that effectively highlights the product's key features."]
    print(f"Loading {args.cvs_file.split('/')[-1]}...")
    # image_files = ["/scratch/js12556/mgllm/9896.jpg"]
    image_files = []
    texts = []
    raw_data = load_dataset("csv", data_files=args.cvs_file)
    # print(raw_data)
    for data in raw_data['train']:
        true_label = {}
        image_files.append(data["image_summary"])
        texts.append(data['text'])
    total_num = len(image_files)
    if total_num != len(texts):
        print(f"Error: total_num{total_num} != len(texts){len(texts)}")
    print("total num of tests:", total_num)
    output_json = []
    for idx in range(args.start_idx, len(image_files)):#len(test_ids)):
        # i = test_ids[idx]
        image_summary_and_description = "Given the information of a product from the Amazon "+args.dataset_name + "dataset: "+str(texts[idx])+". Image summary:\n "+str(image_files[idx])
        i = idx
        query_items = []
        question = prompts[0]
        query_items.append({'text':image_summary_and_description+question})  # Add the question at the end
        query = tokenizer.from_list_format(query_items)
        print("query_items:", query_items)
        # evaluate the model
        output_data = {}
        output_data["id"] = i
        response, _ = model.chat(tokenizer, query=query, history=None)
        output_data["res"] = response
        print("Response: ", response)   
        output_json.append(output_data)
        # save the output
        with open(args.output_file, "w") as file:
                json.dump(output_json, file, indent=4)  # 'indent=4' for pretty-printing
        print(f"[{idx}/{total_num}]")
        if len(output_json) == 1:
            print("Here is the first data to evaluate:")
            print(" query_items:", query_items)
            print(" output_data:", output_data["res"])
            print(" len(question):", len(question))

    print("len(output_json):", len(output_json))