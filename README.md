# Natural Language Processing: Questions Answering
## Summary
This is a natural language processing project on question answering. User can input any question and this algorithm will output the best answer based on the database given.
This question answering is a simple system based on information retrieval.  
All the text files in the corpus are from the Wikipedia website.
The text files are based on the few topics:
1. artificial intelligence
2. machine learning
3. natural language processing
4. neural network
5. probability
6. python

## Technologies used:
```
Python 
Python libraries
NLTK library
```
Algorithm/Formula:
```
Information Retrieval (IR)
Term Frequency Inverse Document Frequency (TF-IDF)
Query Term Density
```

## What is Information Retrieval?
It is a method on finding relevant info in a file or document in response to a user's query. It is using word matching without understand the meaning of those documents about.
Term Frequency Inverse Document Frequency (TF-IDF) = Term Frequency * Inverse Document Frequency
Term Frequency (TF) = Number of times a word/term shows up
Inverse Document Frequency (IDF) = Natural Log (Total documents / Number of Documents Consist the term

Query Term Density is the proportion of terms in the sentence (from passages) given that the term must also appear in the query. 
Query Term Density = number of word in sentence appears in query / total words in sentence 
For example, given that a sentence has 3 words out of 20 appear in the query, then the sentence’s query term density is 0.15, 

## How this project works?
The system will process all the text documents. After a user input a query (in English), information retrieval will first identify which document(s) based on the word matching to the query. The relevant document will be ranked according to the highest TF-IDF. The most relevant document will be subdivided into passages (in this case, sentences). Each sentences will be processed to get the most relevant passage to the query.

## Requirement:
1. Install python3 in Visual Studio Code
2. Install nltk
``pip3 install nltk``
3. Download the whole text package in this master branch
4. Run
``python3 runner.py``
