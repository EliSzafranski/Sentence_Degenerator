import string
import json
import operator
import os
# Get's word count of all words in corpus. Does not include
# numbers or acronyms and such
def get_wc(pathname):
    word_count = dict()
    with open(pathname, 'r') as file:
        for line in file:
            print(line)
            for word in line.split():
                if len(word) > 2 and word.isalpha():
                    print(word)
                    word = word.lower()
                    if word not in word_count:
                        word_count[word] = 1
                    else:
                        word_count[word] += 1
    # DUMP word_count to file
    # print(word_count)
    with open(f"{pathname}.wordcount.json", "w") as file:
        file.write(json.dumps(word_count))
                        
# Returns the top n percent words in the word count and creates a set                      
def get_top_n_percent(percent, wc_json_path):
    # Load word_count file
    word_count = json.load(open(wc_json_path))
    total_words = len(word_count)
    top_np_words = round((total_words*percent) / 100)
    set_of_topn_words = set()
    print(top_np_words)
    # sort dictionary 
    i = 0
    for k,v in sorted(word_count.items(),key=operator.itemgetter(1),reverse=True):
        if i == top_np_words:
            break
        print(k,v)
        set_of_topn_words.add(k)
        i+=1
    with open(f"{wc_json_path[:-14]}top_{percent}_percent_set", "w") as file:
        # dump set into file for later
        file.write(str(set_of_topn_words))

    
# if __name__ == "__main__":
