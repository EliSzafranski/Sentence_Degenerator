from sentence_degenerator import mask_xpercent_words,mask_npercent_new, with_dictionary
from constants import LAN, INPUT_FILE
import json
import pickle
import cProfile, pstats
import concurrent.futures
import time
from itertools import repeat
####################
# BERT Models running locally with saved models
###############

# Pre-trained English BERT model 
# bert_tokenizer = pickle.load(open('../model_caches/bert_tokenizer.sav', 'rb'))
# bert_model = pickle.load(open('../model_caches/bert_model.sav', 'rb'))

# # Multilanguage BERT model
# multi_lang_tokenizer = pickle.load(open('../model_caches/multi_lang_tokenizer.sav', 'rb'))
# multi_lang_model = pickle.load(open('../model_caches/multi_lang_model.sav', 'rb'))

# # Monolingual Spanish BERT model
# spanBert_tokenizer = pickle.load(open('../model_caches/spanBert_tokenizer.sav', 'rb'))
# spanBert_model = pickle.load(open('../model_caches/spanBert_model.sav', 'rb'))

# # Monolingual Hebrew BERT model
# heBert_tokenizer = pickle.load(open('../model_caches/heBert_tokenizer.sav', 'rb'))
# heBert_model = pickle.load(open('../model_caches/heBert_model.sav', 'rb'))

# # Monolingual Russian BET model
# rusBert_tokenizer = pickle.load(open('../model_caches/rusBert_tokenizer.sav', 'rb'))
# rusBert_model = pickle.load(open('../model_caches/rusBert_model.sav', 'rb'))

####################
# BERT Models when running code on the server, downloading the model each time
# (This is meant to save space on the server)
###############
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
# multi_lang_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# multi_lang_model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased').eval()
# spanBert_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
# spanBert_model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
# heBert_tokenizer = BertTokenizer.from_pretrained("avichr/heBERT")
# heBert_model = BertForMaskedLM.from_pretrained("avichr/heBERT")
# rusBert_tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# rusBert_model = BertForMaskedLM.from_pretrained("DeepPavlov/rubert-base-cased")


def damage_corpus(pathname, tokenizer, model):
    #### single threaded version ########
    output_name = pathname[:-7]+".source"
    with open(pathname, 'r') as file, open(output_name, 'w') as output_file:
        next_line = file.readline()
        while next_line != "":
            next_line = f"{mask_npercent_new(next_line, tokenizer, model)}\n"
            output_file.write(next_line)
            next_line = file.readline()
    ##########
    
    ########multi threaded version 
    # entire_text = []
    # with open(pathname) as f:
    #     for line in f:
    #         entire_text.append(line)
            
    # output_name = pathname[:-7]+".source"
    # with open(output_name, 'w') as output_file:
    #         with concurrent.futures.ThreadPoolExecutor() as threads:
    #             for result in threads.map(mask_npercent_new, entire_text, repeat(tokenizer), repeat(model)):
    #                 output_file.write(f"{result}\n")
    
def damage_with_dict(path_to_dict):
    output_name = INPUT_FILE[:-7]+".source"
    big_dictionary = json.load(open(path_to_dict))
    with open(INPUT_FILE, 'r') as file, open (output_name, 'w') as output_file:
        sentence_id = 0
        next_line = file.readline()
        while next_line != '':
            cur_sent_dict = big_dictionary[f'{sentence_id}']
            line = f"{with_dictionary(next_line, cur_sent_dict)}\n"
            output_file.write(line)
            sentence_id+=1
            next_line = file.readline()
    

if __name__ == "__main__":
	ans = input('Do you have a dictionary to use?\tY/n    ')
	if ans.upper() == 'Y':
		path_to_dict = input("Please enter the path to the dictionary: ")
		profiler = cProfile.Profile()
		profiler.enable()
		damage_with_dict(path_to_dict)
		profiler.disable()
		stats = pstats.Stats(profiler)
		# stats.dump_stats('../pstats_out/damage_with_dict.pstats')
		stats = pstats.Stats(profiler).sort_stats('tottime')
		stats.print_stats()
	else:
		langauges = ["EN", "SP", "RUS", 'HEB']
		lan = langauges[LAN-1]
		if lan == 'EN':
			bert_tokenizer = pickle.load(open('../model_caches/bert_tokenizer.sav', 'rb'))
			bert_model = pickle.load(open('../model_caches/bert_model.sav', 'rb'))
		elif lan == "SP":
			bert_tokenizer = pickle.load(open('../model_caches/spanBert_tokenizer.sav', 'rb'))
			bert_model = pickle.load(open('../model_caches/spanBert_model.sav', 'rb'))
		elif lan == "RU":
			bert_tokenizer = pickle.load(open('../model_caches/rusBert_tokenizer.sav', 'rb'))
			bert_model = pickle.load(open('../model_caches/rusBert_model.sav', 'rb'))
		else:
			bert_tokenizer = pickle.load(open('../model_caches/heBert_tokenizer.sav', 'rb'))
			bert_model = pickle.load(open('../model_caches/heBert_model.sav', 'rb'))
		# profiler = cProfile.Profile()
		# profiler.enable()
		start_time = time.perf_counter()
		damage_corpus(INPUT_FILE, bert_tokenizer, bert_model)
		print("--- %s seconds ---" % (time.perf_counter() - start_time))
		# profiler.disable()
		# stats = pstats.Stats(profiler)
		# # stats.dump_stats('../pstats_out/old_old_func.pstats')
		# stats = pstats.Stats(profiler).sort_stats('tottime')
		# stats.print_stats()
	
 
# damage_corpus('./sentences.txt')
# print(CORUPT_TOKENS_PERCENTAGE)

