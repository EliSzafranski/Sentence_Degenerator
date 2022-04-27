import string

TOP_CLEAN 				 = int(input("Enter the TOP_CLEAN: "))
TOP_K    				 = int(input("Enter the TOP_K: "))
CORUPT_TOKENS_PERCENTAGE = float(input("Enter the CORUPT_TOKENS_PERCENTAGE: "))
LAN = int(input("Enter a number to choose a Language:\n[1]\tEN\n[2]\tSP\n[3]\tRUS\n[4]\tHEB\n"))
INPUT_FILE = input("Enter the name of the file you would like to corrupt: ")
IGNORE_TOKENS = set(string.punctuation)
IGNORE_TOKENS.update(['[PAD]', '[UNK]', '...'])