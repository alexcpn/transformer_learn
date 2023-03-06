"""
Helper to clean up the downloaded file
 original data https://gdlp01.c-wss.com/gds/0/0300004730/02/eosrt3-eos1100d-im2-c-en.pdf
"""
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Clean up the data
def cleanFile(fileNameIn, fileNameOut):
   
    """ Removes a given line from a file """
    with open(fileNameIn, 'r') as read_file:
        lines = read_file.readlines()

    with open(fileNameOut, 'w') as write_file:
        for line in lines:
            if len(line) < 7:
                pass
            else:
                tokens = word_tokenize(line)
                words = [word for word in tokens if word.isalpha() and len(word)>1]
                write_file.write(' '.join(map(str, words))+ " " )


cleanFile("./data/eosrt3-eos1100d-im2-c-en.txt", "./data/clean2.txt")