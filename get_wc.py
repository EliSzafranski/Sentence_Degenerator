import json
import operator
# Get's word count of all words in corpus. Does not include
# numbers or acronyms and such
def get_wc(pathname):
    word_count = dict()
    with open(pathname, 'r') as file:
        for line in file:
            # print(line)
            for word in line.split():
                if len(word) > 2 and word.isalpha():
                    # print(word)
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
def get_top_n_percent(percent, wc_json_path=''):
    # Load word_count file
    if __name__ != "__main__":
        ans = input("Do you have a word_count file yet? y/[N]\t")
        ans = ans.lower()
        if ans == 'y':
            wc_json_path = input("Enter the path to the word count file: ")
        else:
            target_file = input("Enter the path to the file you would like to get a word count of: ") 
            get_wc(target_file)
            wc_json_path = f"{target_file}.wordcount.json"
            
    word_count = json.load(open(wc_json_path))
    total_words = len(word_count)
    top_np_words = round((total_words*percent) / 100)
    set_of_topn_words = set()
    # print(top_np_words)
    # sort dictionary 
    i = 0
    for k,v in sorted(word_count.items(),key=operator.itemgetter(1),reverse=True):
        if i == top_np_words:
            break
        # print(k,v)
        set_of_topn_words.add(k)
        i+=1
    with open(f"{wc_json_path[:-14]}top_{percent}_percent_set", "w") as file:
        # dump set into file for later
        file.write(str(set_of_topn_words))

    
if __name__ == "__main__":
    init_ans = input("Do you have a word_count yet? y/[N]\t")
    init_ans = init_ans.lower()
    if init_ans == 'y':
        wcp = input("Enter the path to the word count file: ")
        np = int(input("What percentage of the words would you like?  "))
        get_top_n_percent(wcp, np)
    else:
        p = input("Enter the path to the file you would like to get a word count of: ")
        get_wc(p)
        nip = input("Would you like to create a set of the top occuring words? [Y]/n\t")
        if nip == 'N' or nip == 'n':
            pass
        else:
            tnp = int(input("What percentage of the words would you like?  "))
            get_top_n_percent(f"{p}.wordcount.json", tnp)