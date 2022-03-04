# python3 questions.py corpus

import math
import nltk
import sys
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")
    
    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_directory = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            files_directory[filename] = f.read()
            
    return files_directory


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    return list(
        word.lower() for word in nltk.word_tokenize(document) 
        if word.lower() not in string.punctuation
        and word.lower() not in nltk.corpus.stopwords.words("english")
        )


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.5
    """
    # ensure every word is unique
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # apply idf formula = total files in document / total document contain word
    word_appear = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        word_appear[word] = math.log(len(documents) / f)

    return word_appear


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """    
    # if the word in query can be found in files in document, compute the frequency
    # frequency will be sum up according the file
    # tfidfs for each word = term frequency * idfs
    # sum up all tfidfs for a file
    tfidfs = dict()
    for filename in files:
        tfidfs[filename] = 0
        frequency = dict()
        for input_word in query:
            for word in files[filename]:   
                if input_word == word:
                    if input_word not in frequency:
                        frequency[input_word] = 1
                    frequency[input_word] += 1
            if input_word in frequency:
                calculation =  frequency[input_word] * idfs[input_word]
                tfidfs[filename] += calculation

    # sort the tfidfs value for each file descendingly
    rank = [k for k, v in sorted(tfidfs.items(), key=lambda item: item[1], reverse=True)]
    
    return [rank[n-1]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # check how many unique word in the query can be in the sentence
    # if exist, sum up the idfs of the word 
    sum_idfs = dict()
    for sentence in sentences:
        sum_idfs[sentence] = 0
        matching_word = []
        for input_word in query:
            for word in sentences[sentence]:   
                if input_word == word and word not in matching_word:
                    sum_idfs[sentence] += idfs[word]
                    matching_word.append(word)
    
    # sort the idfs of words
    rank = [k for k, v in sorted(sum_idfs.items(), key=lambda item: item[1], reverse=True)]
    # print(rank)
    # print(rank[0], sum_idfs[rank[0]])
    # print(rank[1], sum_idfs[rank[1]])

    # if any 2 of the idfs are same, check for query term density
    # query term density = length of the word in sentence / how many sentence's words match with the word in query 
    # rank the query term density descendingly
    variable = 0
    for number in range(1, len(rank)):
        value = sum_idfs[rank[variable]]
        next_value = sum_idfs[rank[number]]

        if value == next_value:

            frequency = dict()

            frequency[rank[variable]] = 0
            for word in sentences[rank[variable]]:
                if word in query:
                    frequency[rank[variable]] += 1
            frequency[rank[variable]] = frequency[rank[variable]] / len(rank[variable])

            frequency[rank[number]] = 0
            for word in sentences[rank[number]]:
                if word in query:
                    frequency[rank[number]] += 1
            frequency[rank[number]] = frequency[rank[number]] / len(rank[number])

            if frequency[rank[variable]] < frequency[rank[number]]:
                rank[variable] = rank[number]
                rank[number] = rank[variable]

        variable += 1          

    return [rank[n-1]]

if __name__ == "__main__":
    main()
