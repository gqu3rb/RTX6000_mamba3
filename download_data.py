from datasets import load_dataset
import os

# The directory path set in your llm.yaml
data_dir = "/home/ZhongLongYou/workspace/mamba-main/dataset/tiny_stories"
os.makedirs(data_dir, exist_ok=True)

print("Downloading TinyStories dataset from Hugging Face... (This may take a while)")
dataset = load_dataset("roneneldan/TinyStories")

print("Writing train.txt ...")
with open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        f.write(item["text"] + "\n<|endoftext|>\n")

print("Writing valid.txt ...")
with open(os.path.join(data_dir, "valid.txt"), "w", encoding="utf-8") as f:
    for item in dataset["validation"]:
        f.write(item["text"] + "\n<|endoftext|>\n")

print("Writing test.txt (copied from validation set to prevent errors) ...")
with open(os.path.join(data_dir, "test.txt"), "w", encoding="utf-8") as f:
    for item in dataset["validation"]:
        f.write(item["text"] + "\n<|endoftext|>\n")

print("Dataset preparation complete!")