from hashlib import new
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random
import string
import pandas as pd
import numpy as np
import math
import pickle
import cProfile,pstats

from constants import TOP_CLEAN, TOP_K, CORUPT_TOKENS_PERCENTAGE, IGNORE_TOKENS


# Given a tokenized sentence, turn it to a human readable form 
def decode(tokenizer, pred_idx, top_clean):
    # Ideas to speed up: Turn this into a set, and maybe have it as a constant
    ######## PRE ########
    # ignore_tokens = string.punctuation + '[PAD]' + '...' + '[UNK]'
    # tokens = []
    # for w in pred_idx:
    #     # map id of predicted word to token. Function returns string of characters seperated by spaces
    #     interim = tokenizer.decode(w)
    #     # clean the deocded value by splitting by spaces and joining to one big string
    #     token = ''.join(interim.split())
    #     if token not in ignore_tokens:
    #         # more cleaning of decoded value, make sure token isn't punctuation
    #         tokens.append(token.replace('##', ''))
    # # of all the decoded guesses, only return the top_clean ones
    # return '\n'.join(tokens[:top_clean])
    #################################################
    
    ######### POST ########
    tokens = []
    for w in pred_idx:
        # map id of predicted word to token. Function returns string of characters seperated by spaces
        interim = tokenizer.decode(w)
        # clean the deocded value by splitting by spaces and joining to one big string
        token = ''.join(interim.split())
        if token not in IGNORE_TOKENS:
            # more cleaning of decoded value, make sure token isn't punctuation
            tokens.append(token.replace('##', ''))
    # of all the decoded guesses, only return the top_clean ones
    return '\n'.join(tokens[:top_clean])
    #######################
    
    
#######################################################################

# given a sentence, tokenize it accoding to the BERT tokenizer rules 
def map_to_id(tokenizer, text_sentence, add_special_tokens=True):
    
    text_sentence = text_sentence.replace('<TO_REPLACE>', tokenizer.mask_token)
    # second way to do the above: 
#     rplcmt_idx = text_sentence.split().index('<TO_REPLACE>')
#     text_sentence = text_sentence.split()
#     text_sentence[rplcmt_idx] = tokenizer.mask_token
#     text_sentence = ' '.join(text_sentence)
    
    # use the bert tokenizer to split sentence into tokens 
    tokenized_sentence = tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)
    
    # cast the tokenized sentence id's to a tensor (ie a list)
    input_ids = torch.tensor([tokenized_sentence])
#     print(input_ids)

    # The value of bert_tokenizor.mask_token_id is 103, so store the index of of the first
    # instance of 103 wherever it appears in our tensor
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
#     print(mask_idx)
    return input_ids, mask_idx

# Take a sentence with a <TO_REPLACE> token, and return BERTs predictions for that word
def run_bert(sentence, tokenizer, model):
    # Experiment with changing these params 
    # Tokenize the sentence and get id's according to the BERT tokenizer
    input_ids, mask_idx = map_to_id(tokenizer, sentence)
    
    ##### This part right here takes the most time ##############
    with torch.no_grad():
        # Run the BERT model and predictions based on given vocab id's
        predict = model(input_ids)[0]
    #############################################################
    
    bert = decode(tokenizer, predict[0, mask_idx, :].topk(TOP_K).indices.tolist(), TOP_CLEAN)
    return {'bert': bert}


# choose a random word in the sentence to predict
def choose_word_to_predict(sentence):
    # Ideas to speed up: Turn this into a set, and maybe have it as a constant, same constant as above
    ##### PRE
    # symbols = set(string.punctuation)
    ########
    
    words = sentence.split()
    idx_to_replace = random.randint(0, len(words)-1)
    ####### PRE
    #while words[idx_to_replace] in symbols:
    ################
    while words[idx_to_replace] in IGNORE_TOKENS:   
        idx_to_replace = random.randint(0, len(words)-1)
    words[idx_to_replace] = '<TO_REPLACE>'
    return ' '.join(words),idx_to_replace


# Return at most the top K clean words BERT thinks the word could be for any sentence
def get_word_predictions(input_sentence, tokenizer, model):
    padded_sentence,replaced_idx = choose_word_to_predict(input_sentence)
#     padded_sentence,replaced_idx = mask_xpercent_words(input_sentence)
    
    results = run_bert(padded_sentence, tokenizer, model)
    return results,padded_sentence,replaced_idx

# Take in a sentence, and return the sentence with one word swapped out with a BERT prediction
def change_up_sentence(correct_sentence, tokenizer, model):
    res,ps,replaced_idx = get_word_predictions(correct_sentence, tokenizer, model)
    top_word_predictions = res['bert'].split("\n")
    new_sentence = correct_sentence.split()
    ##### print(f'Input sentence: {correct_sentence}')
    ##### print(f'Word to change: {ps}')
    ##### print(f'Possible words: {top_word_predictions}')
    # It should be a random word, not necessarily the last element; set this param
    # new_sentence[replaced_idx] = top_word_predictions[-1]
    
    # Choose random index in range of 0 to len(top_word_predictions) for the new word
    new_word_idx = random.randrange(len(top_word_predictions)) - 1
    ##### print(f'Change [{new_sentence[replaced_idx]}] to [{top_word_predictions[new_word_idx]}] (idx: {new_word_idx} out of {len(top_word_predictions)-1})')
    new_sentence[replaced_idx] = top_word_predictions[new_word_idx]
    
    ##### print(f"New sentence: {' '.join(new_sentence)}")
    return ' '.join(new_sentence)


# Take in a sentence, and return the sentence with one word swapped out with a BERT prediction
def change_specific_word(correct_sentence, tokenizer, model, replaced_word_idx=0):
    ########## PRE
    # symbols = set(string.punctuation)
    ##################
    words = correct_sentence.split()
    ########## PRE
    #while words[replaced_word_idx] in symbols:
    #########
    while words[replaced_word_idx] in IGNORE_TOKENS:
        replaced_word_idx = (replaced_word_idx + 1) % len(words)
    words[replaced_word_idx] = '<TO_REPLACE>'
    padded_sentence = ' '.join(words)
    
    results = run_bert(padded_sentence, tokenizer, model)
    top_word_predictions = results['bert'].split("\n")
    new_sentence = correct_sentence.split()
    
    ##### print(f'Input sentence: {correct_sentence}')
    ##### print(f'Word to change: {padded_sentence}')
    ##### print(f'Possible words: {top_word_predictions}')
    
    # It should be a random word, not necessarily the last element; set this param
    # new_sentence[replaced_idx] = top_word_predictions[-1]
    
    # Choose random index in range of 0 to len(top_word_predictions) for the new word
    new_word_idx = random.randrange(len(top_word_predictions)) - 1
    ##### print(f'Change [{new_sentence[replaced_word_idx]}] to [{top_word_predictions[new_word_idx]}] (idx: {new_word_idx} out of {len(top_word_predictions)-1})')
    new_sentence[replaced_word_idx] = top_word_predictions[new_word_idx]
    
    ##### print(f"New sentence: {' '.join(new_sentence)}")
    return ' '.join(new_sentence)



# Takes in a sentence, creates masks for X percent of the words, and then returns the predictions of those words
def mask_xpercent_words(sentence, tokenizer, model, idxs_to_mask=None):
    words = sentence.split()
    num_words_to_mask = math.floor(len(words) * CORUPT_TOKENS_PERCENTAGE)
    num_words_to_mask = 1 if num_words_to_mask == 0 else num_words_to_mask
    idxs_to_mask = random.sample(range(0, len(words)), num_words_to_mask) if idxs_to_mask is None else idxs_to_mask
    ##### print(f"Initial indices to change: {idxs_to_mask}\n")
    for i in idxs_to_mask:
        sentence = change_specific_word(sentence, tokenizer, model, i)
        ##### print('***'*10)

    return sentence



################################
# Testing functions
def reproducable_test_function(correct_sentence, tokenizer, model, idx_to_replace=0):
    words = correct_sentence.split()
    words[idx_to_replace] = '<TO_REPLACE>'
    padded_sentence = ' '.join(words)
    
    results = run_bert(padded_sentence, tokenizer, model)
    top_word_predictions = results['bert'].split("\n")
    
    # print(f'Input sentence: {correct_sentence}')
    # print(f'Word to change: {padded_sentence}')
    # print(f'Possible words: {top_word_predictions}')
    
    # It should be a random word, not necessarily the last element; set this param
    # new_sentence[replaced_idx] = top_word_predictions[-1]
    # Choose random index in range of 0 to len(top_word_predictions) for the new word
    new_word_idx = random.randrange(len(top_word_predictions)) - 1
    # print(f'Change [{correct_sentence.split()[idx_to_replace]}] to [{top_word_predictions[new_word_idx]}] (idx: {new_word_idx} out of {len(top_word_predictions)-1})')
    words[idx_to_replace] = top_word_predictions[new_word_idx]
    # print(f"New sentence: {' '.join(words)}")
    return ' '.join(words)


def test1(btkz, bm):
    print('*' * 20)
    print("Test 1: Using mask_xpercent_words function\n")
    mask_xpercent_words('There is a need to generate random numbers when studying a model or behavior of a program for different range of values', btkz, bm)
    print('*' * 20)

######################################################################


# Function to place mask tokens in sentence
def place_masks(orig_sentence, tokenizer):
    orig_sentence = orig_sentence.split()
    num_words_to_mask = round(len(orig_sentence) * CORUPT_TOKENS_PERCENTAGE)
    num_words_to_mask = num_words_to_mask if num_words_to_mask > 0 else 1
    rnsmp = random.sample(range(0, len(orig_sentence)), num_words_to_mask)
    # Ensures mask is not a punctuation 
    for i in range(len(rnsmp)):
        counter = 0
        while orig_sentence[rnsmp[i]] in IGNORE_TOKENS:
            rnsmp[i] = (rnsmp[i] + 1) % len(orig_sentence)
            counter += 1
            if counter >= len(orig_sentence):
                print(' '.join(orig_sentence))
                break
        orig_sentence[rnsmp[i]] = tokenizer.mask_token
    rnsmp.sort()
    return rnsmp,' '.join(orig_sentence)

# takes in the tokenized predictions for all the masked words of the model at once, and decodes them
def new_decode(tokenizer, tokenized_prediction):
    all_words = []
    for i in range(len(tokenized_prediction)):
        tokens = []
        for w in tokenized_prediction[i]:
            interim = tokenizer.decode(w)
            # clean the deocded value by splitting by spaces and joining to one big string
            tkn = ''.join(interim.split())
            if tkn not in IGNORE_TOKENS:
                # more cleaning of decoded value, make sure token isn't punctuation or weird token
                tokens.append(tkn.replace('##', ''))
        # only return the TOP_CLEAN values
        all_words.append(tokens[:TOP_CLEAN].copy())
        tokens.clear()
    return all_words

# Replaces the masked word in the sentence with a random predicted word from TOP_CLEAN predictions
def replace_word(top_word_predictions, idx_to_replace, orig_sentence):
    new_sentence = orig_sentence.split()
    # Choose random index in range of 0 to len(top_word_predictions) for the new word
    new_word_idx = random.randrange(len(top_word_predictions)) - 1
#     print(f'Change [{new_sentence[idx_to_replace]}] to [{top_word_predictions[new_word_idx]}] (idx: {new_word_idx} out of {len(top_word_predictions)-1})')
    new_sentence[idx_to_replace] = top_word_predictions[new_word_idx]
    ##### print(f"New sentence: {' '.join(new_sentence)}")
    return ' '.join(new_sentence)

def mask_npercent_new(some_sentence, tokenizer, model):
    replaced_indexs,snt = place_masks(some_sentence, tokenizer)
    # Tokenizes the input sentence
    tokenized_sentence = tokenizer(snt, return_tensors='pt')
#     print(tokenized_sentence)
    # Returns the indexes of the mask tokens
    mask_idx = torch.where(tokenized_sentence['input_ids'] == tokenizer.mask_token_id)[1].tolist()
    
    # Gets the predictions of all the masked words
    model_uncleaned_output = model(**tokenized_sentence)
#     print(mask_idx)

    # return top k predictions per word
    tokenized_predictions = model_uncleaned_output[0][0, mask_idx, :].topk(TOP_K).indices.tolist()
#     print(tokenized_predictions)
    decoded_preds = new_decode(tokenizer, tokenized_predictions)
#     print(decoded_preds)
    for i in range(len(decoded_preds)):
        some_sentence = replace_word(decoded_preds[i], replaced_indexs[i], some_sentence)
    return some_sentence

################ With a dictionary ###################
# Function to place mask tokens in sentence
def get_mask_idxs(orig_sentence, sent_dict):
    orig_sentence = orig_sentence.split()
    num_words_to_mask = round(len(orig_sentence) * CORUPT_TOKENS_PERCENTAGE)
    rnsmp = random.sample(range(0, len(orig_sentence)), num_words_to_mask)
    # Ensures mask is not a punctuation 
    for i in range(len(rnsmp)):
        counter = 0
        while str(rnsmp[i]) not in sent_dict:
            rnsmp[i] = (rnsmp[i] + 1) % len(orig_sentence)
            counter += 1
            if counter >= len(orig_sentence):
                print(' '.join(orig_sentence))
                return []
    return rnsmp

def with_dictionary(sentence, sent_dict):
    mask_idxs = get_mask_idxs(sentence, sent_dict)
    if len(mask_idxs) == 0:
        return sentence
    new_sentence = sentence.split()
    for idx in mask_idxs:
        list_of_preds = sent_dict[f'{idx}']
        new_sentence[idx] = list_of_preds[random.randint(0, TOP_CLEAN-1)]
    return ' '.join(new_sentence)
    
    
    
if __name__ == "__main__":
    # Default English languge model
    # Pre-trained English BERT model 
    bert_tokenizer = pickle.load(open('./model_caches/bert_tokenizer.sav', 'rb'))
    bert_model = pickle.load(open('./model_caches/bert_model.sav', 'rb'))
    test1(bert_tokenizer, bert_model)
    # test2()
