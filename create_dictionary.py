import json
import torch
import pickle
import string
import os
import time
from os.path import exists
from get_wc import get_top_n_percent
import ast
IGNORE_TOKENS = set(string.punctuation)
IGNORE_TOKENS.update(['[PAD]', '[UNK]', '...'])


################################################################################
def new_decode_pt2(tokenizer, tokenized_prediction):
    tokens = []
    for w in tokenized_prediction:
        interim = tokenizer.decode(w)
        # clean the deocded value by splitting by spaces and joining to one big string
        tkn = ''.join(interim.split())
        # if tkn not in IGNORE_TOKENS:
        #     # more cleaning of decoded value, make sure token isn't punctuation
        #     tokens.append(tkn.replace('##', ''))
        tokens.append(tkn.replace('##', ''))
    return tokens

def get_word_predictions(sentence, index_of_mask, tokenizer, model):
    sentence[index_of_mask] = tokenizer.mask_token
    # print(' '.join(sentence))
    tokenized_sentence = tokenizer(' '.join(sentence), return_tensors='pt')
#     print(tokenized_sentence)
    mask_idx = torch.where(tokenized_sentence['input_ids'] == tokenizer.mask_token_id)[1].tolist()
#     print(mask_idx)
    model_uncleaned_output = model(**tokenized_sentence)
#     print(model_uncleaned_output)
    tokenized_predictions = model_uncleaned_output[0][0, mask_idx, :].topk(150).indices.tolist()[0]
#     print(tokenized_predictions)
    decoded_preds = new_decode_pt2(tokenizer, tokenized_predictions)
    # print(decoded_preds)
    return decoded_preds

def get_sentence_dict(some_sentence, tokenizer, model):
    sent_dict = {}
    idx = 0
    sent_list = some_sentence.split()
    sent_len = len(sent_list)
    while idx < sent_len:
        if sent_list[idx] not in set_of_allowed_words:
        # if sent_list[idx] in IGNORE_TOKENS:
            idx += 1
            continue
        sent_dict[idx] = get_word_predictions(sent_list.copy(), idx, tokenizer, model)
        idx+=1
    return sent_dict

def save_dict_to_file(pathname, tokenizer, model, path_to_set):
    global set_of_allowed_words 
    with open(path_to_set, 'r') as f:
        set_of_allowed_words = ast.literal_eval(f.read())
    num_lines = os.popen(f"wc -l {pathname}").read()
    num_lines = [int(s) for s in num_lines.split() if s.isdigit()][0]
    # num_lines = num_lines[5:5+num_lines[5:].index(' ')]
    with open(pathname, 'r') as file, open (f"{pathname}.dict", 'w') as output_file:
        output_file.write('{')
        next_line = file.readline()
        output_file.write(f'"0":{json.dumps(get_sentence_dict(next_line, tokenizer, model))}')
        next_line = file.readline()
        i = 1
        while next_line != "":
            output_file.write(',\n')
            line = f'"{i}":{json.dumps(get_sentence_dict(next_line, tokenizer, model))}'
            output_file.write(line)
            if i % 10 == 0:
                print(f"{i}/{num_lines} completed")
            i+=1
            next_line = file.readline()
        output_file.write('}')


if __name__ == '__main__':
    input_file = input("Enter the name of the file you would like to turn into a dictionary: ")
    lan = int(input("Enter a number to choose a Language:\n[1]\tEN\n[2]\tSP\n[3]\tRUS\n[4]\tHEB\n"))
    if lan == 1:
        bert_tokenizer = pickle.load(open('../model_caches/bert_tokenizer.sav', 'rb'))
        bert_model = pickle.load(open('../model_caches/bert_model.sav', 'rb'))
    elif lan == 2:
        bert_tokenizer = pickle.load(open('../model_caches/spanBert_tokenizer.sav', 'rb'))
        bert_model = pickle.load(open('../model_caches/spanBert_model.sav', 'rb'))
    elif lan == 3:
        bert_tokenizer = pickle.load(open('../model_caches/rusBert_tokenizer.sav', 'rb'))
        bert_model = pickle.load(open('../model_caches/rusBert_model.sav', 'rb'))
    else:
        bert_tokenizer = pickle.load(open('../model_caches/heBert_tokenizer.sav', 'rb'))
        bert_model = pickle.load(open('../model_caches/heBert_model.sav', 'rb'))

    set_exists = input("Do you have a set of words to use? Y/n ")
    set_exists = set_exists.lower()
    if set_exists == 'y':
        path_to_set = input("Enter the path to the set: ")
    else:
        pot = int(input("What percentage of the words would you like?  "))
        get_top_n_percent(pot)
        path_to_set = f"{input_file}.top_{pot}_percent_set"
        
    start_time = time.time()
    save_dict_to_file(input_file, bert_tokenizer, bert_model, path_to_set)
    print("--- %s seconds ---" % (time.time() - start_time))