import string
import torch
import subprocess
from get_wc import get_top_n_percent

TOP_CLEAN 				 = int(input("Enter the TOP_CLEAN: "))
TOP_K    				 = int(input("Enter the TOP_K: "))
CORUPT_TOKENS_PERCENTAGE = float(input("Enter the CORUPT_TOKENS_PERCENTAGE: "))
LAN = int(input("Enter a number to choose a Language:\n[1]\tEN\n[2]\tSP\n[3]\tRUS\n[4]\tHEB\n"))
INPUT_FILE = input("Enter the name of the file you would like to corrupt: ")
IGNORE_TOKENS = set(string.punctuation)
IGNORE_TOKENS.update(['[PAD]', '[UNK]', '...'])
if torch.cuda.is_available():
    subprocess.run(["nvidia-smi"])
    device_num = int(input("Which GPU would you like to use? "))
    dev = f"cuda:{device_num}"
else:
	dev = "cpu"

DEVICE = torch.device(dev)

init_ans = input("Do you only want to predict only the most frequent words? Y/n ")
if init_ans.lower() == 'y':
    next_ans = input("Do you already have a set of words? Y/n ")
    if next_ans.lower() == 'n':
        percent = int(input("What percent of words would you like to check? "))
        get_top_n_percent(percent)
        SET_OF_WORDS = f"{INPUT_FILE}.top_{percent}_percent_set"
    else:
        SET_OF_WORDS = input("Enter the path to the set: ")
    