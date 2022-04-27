from sentence_degenerator import mask_xpercent_words,mask_npercent_new
from constants import LAN, INPUT_FILE
import in_place
import pickle
import cProfile, pstats
####################
# BERT Models running locally with saved models
###############

# Pre-trained English BERT model 
bert_tokenizer = pickle.load(open('./model_caches/bert_tokenizer.sav', 'rb'))
bert_model = pickle.load(open('./model_caches/bert_model.sav', 'rb'))

# Multilanguage BERT model
multi_lang_tokenizer = pickle.load(open('./model_caches/multi_lang_tokenizer.sav', 'rb'))
multi_lang_model = pickle.load(open('./model_caches/multi_lang_model.sav', 'rb'))

# Monolingual Spanish BERT model
spanBert_tokenizer = pickle.load(open('./model_caches/spanBert_tokenizer.sav', 'rb'))
spanBert_model = pickle.load(open('./model_caches/spanBert_model.sav', 'rb'))

# Monolingual Hebrew BERT model
heBert_tokenizer = pickle.load(open('./model_caches/heBert_tokenizer.sav', 'rb'))
heBert_model = pickle.load(open('./model_caches/heBert_model.sav', 'rb'))

# Monolingual Russian BET model
rusBert_tokenizer = pickle.load(open('./model_caches/rusBert_tokenizer.sav', 'rb'))
rusBert_model = pickle.load(open('./model_caches/rusBert_model.sav', 'rb'))

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


def damage_corpus(pathname, output_name, language="EN"):
	if language == "HEB":
		with open(pathname, 'r') as file, open(output_name, 'w') as output_file:
			# for line in file:
			# 	line = f"{mask_xpercent_words(line, heBert_tokenizer, heBert_model)}\n"
   			#	output_file.write(line)
			output_file.write(line)
			next_line = file.readline()
			while next_line != "":
				next_line = line = f"{mask_xpercent_words(next_line, heBert_tokenizer, heBert_model)}\n"
				output_file.write(next_line)
				next_line = file.readline()

	elif language == "SP":
		with open(pathname, 'r') as file, open(output_name, 'w') as output_file:
			for line in file:
				line = f"{mask_xpercent_words(line, spanBert_tokenizer, spanBert_model)}\n"
				output_file.write(line)
	elif language == "RUS":
		with open(pathname, 'r') as file, open(output_name, 'w') as output_file:
			for line in file:
				line = f"{mask_xpercent_words(line, rusBert_tokenizer, rusBert_model)}\n"
				output_file.write(line)
	else:
		with open(pathname, 'r') as file, open(output_name, 'w') as output_file:
			# for line in file:
			# 	line = f"{mask_xpercent_words(line, bert_tokenizer, bert_model)}\n"
			# 	output_file.write(line)
			next_line = file.readline()
			while next_line != "":
				# next_line = f"{mask_xpercent_words(next_line, bert_tokenizer, bert_model)}\n"
				next_line = f"{mask_npercent_new(next_line,bert_tokenizer, bert_model)}\n"
				output_file.write(next_line)
				next_line = file.readline()

if __name__ == "__main__":
	langauges = ["EN", "SP", "RUS", 'HEB']
	lan = langauges[LAN-1]
	oFile = input("What would you like to call the output file: ")
 
	# profiler = cProfile.Profile()
	# profiler.enable()
	damage_corpus(INPUT_FILE,oFile, lan)
	# profiler.disable()
	# stats = pstats.Stats(profiler)
	# stats.dump_stats('new_func4.pstats')
	# stats = pstats.Stats(profiler).sort_stats('tottime')
	# stats.print_stats()
	# damage_corpus('simple_test.txt', 'simple_test.txt.source', "EN")
	
 
# damage_corpus('./sentences.txt')
# print(CORUPT_TOKENS_PERCENTAGE)

