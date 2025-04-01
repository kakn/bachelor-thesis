import re
import string
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def testdata(gen, big=False, test=False):
    lines = open('/Users/kasper/bojack/sem6/thesis/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    conversations = open('/Users/kasper/bojack/sem6/thesis/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    metadata = open('/Users/kasper/bojack/sem6/thesis/movie_titles_metadata.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
    swtor = ["m3", "m5", "m34", "m125", "m337", "m489", "m529", "m531", "m328", "m253", "m433"]
    genres = ["'thriller'", "'comedy'", "'horror'", "'drama'"]
    movid = []
    for line in metadata[:-1]:
        _line = line.split(' +++$+++ ')
        genre = _line[-1].split(',')
        if len(genre) == 1:
            p = genre[0].strip("").strip('[').strip(']')
            if gen == p:
                movid.append(_line[0])
            if gen == "'sci-fi'":
                for s in swtor:
                    movid.append(s)
            if gen == "'combo'":
                if p in genres:
                    movid.append(_line[0])
                for s in swtor:
                    movid.append(s)
        if big == True:
            if gen == "'horror'" or gen == "'combo'":
                if len(genre) == 2:
                    for i in range(2):
                        p = genre[i].strip("").strip('[').strip(']')
                        if p == "'horror'":
                            if _line[0] not in movid:
                                movid.append(_line[0])
    id2line = {}
    pines = []
    ponversations = []
    for line in lines[:-1]:
        _line = line.split(' +++$+++ ')
        if _line[2] in movid:
            pines.append(line)
    for conv in conversations[:-1]:
        _conv = conv.split(' +++$+++ ')
        if _conv[2] in movid:
            ponversations.append(conv)
    lines = pines
    conversations = ponversations
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    # Creating a list of all conversations
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))
    # Separating questions and answers
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i+1]])
    pairs = preproc(questions, answers)
    if test == True:
        if big == True:
            return pairs[2000:2231]
        else:
            return pairs[1000:1231]
    else:
        if big == True:
            return pairs[:2000]
        else:
            return pairs[:1000]

# Cleaning texts

def clean_text(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# Necessary preprocessing

def preproc(questions, answers):
    # Cleaning the questions
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))
    # Cleaning the answers
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # Filtering out the questions and answers that are too short or too long
    short_questions = []
    short_answers = []
    i = 0
    for question in clean_questions:
        if 2 <= len(question.split()) <= 25:
            short_questions.append(question)
            short_answers.append(clean_answers[i])
        i += 1
    clean_questions = []
    clean_answers = []
    i = 0
    for answer in short_answers:
        if 2 <= len(answer.split()) <= 25:
            clean_answers.append(answer)
            clean_questions.append(short_questions[i])
        i += 1

    # Combining Q&A's for next step
    pairs = []
    for i in range(len(clean_questions)):
        pairs.append([clean_questions[i], clean_answers[i]])
    random.seed(42)
    random.shuffle(pairs)
    #print(len(pairs))
    return pairs

# Pre-vectorization step

def prevec(pairs):
    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()
    both_tokens = set()

    for line in pairs:
      input_doc, target_doc = line[0], line[1]
      # Appending each input sentence to input_docs
      input_docs.append(input_doc)
      # Splitting words from punctuation
      target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
      # Redefine target_doc below and append it to target_docs
      target_doc = '<START> ' + target_doc + ' <END>'
      target_docs.append(target_doc)

      # Now we split up each sentence into words and add each unique word to our vocabulary set
      for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
          input_tokens.add(token)
        if token not in both_tokens:
            both_tokens.add(token)
      for token in target_doc.split():
        if token not in target_tokens:
          target_tokens.add(token)
        if token not in both_tokens:
            both_tokens.add(token)

    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    global num_encoder_tokens
    global num_decoder_tokens
    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)
    both_tokens_len = len(both_tokens)

    return both_tokens_len

def bain():
    genz = ["'combo'", "'comedy'", "'drama'", "'horror'", "'thriller'", "'sci-fi'"]
    rez = []
    rez2 = []
    for gen in genz:
        pairs = testdata(gen, test=True)
        toks = prevec(pairs)
        rez.append(toks)
    for gen in genz:
        pairs = testdata(gen, big=True, test=True)
        toks = prevec(pairs)
        rez2.append(toks)
    width = 0.40
    x = np.arange(6)
    plt.bar(x-0.2, rez2, width, color='gray')
    plt.bar(x+0.2, rez, width, color='lightgray')
    plt.xticks(x, ['General', 'Comedy', 'Drama', 'Horror', 'Thriller', 'Sci-fi'])
    plt.xlabel("Test data")
    plt.ylabel("Unique tokens")
    plt.legend(["Large", "Small"])
    plt.show()

def main():
    genz = ["'combo'", "'comedy'", "'drama'", "'horror'", "'thriller'", "'sci-fi'"]
    rez = []
    rez2 = []
    for gen in genz:
        pairs = testdata(gen)
        toks = prevec(pairs)
        rez.append(toks)
    for gen in genz:
        pairs = testdata(gen, big=True)
        toks = prevec(pairs)
        rez2.append(toks)
    width = 0.40
    x = np.arange(6)
    plt.bar(x-0.2, rez2, width, color='gray')
    plt.bar(x+0.2, rez, width, color='lightgray')
    plt.xticks(x, ['General', 'Comedy', 'Drama', 'Horror', 'Thriller', 'Sci-fi'])
    plt.xlabel("Training data")
    plt.ylabel("Unique tokens")
    plt.legend(["Large", "Small"])
    plt.show()
bain()
