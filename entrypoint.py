from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
from collections import Counter
from itertools import islice
import pickle
import spacy
import json
import os
import sys

# Load a pickle file


def load_pickle(filename, type):
    with open(filename, type) as f:
        data = pickle.load(f)
    return data


# Reads all letters and fileanmes from specified directory (incl. spaCy initialization)
def get_letters_incl_spacy(directory, filetype):
    letters = []
    for filename in os.listdir(directory):
        if filename.endswith(filetype):
            letters.append([filename, nlp(open(directory + filename).read())])
    return letters


# Gets individual sentences from each letter
def get_sentences(letters):
    sentences = []
    for letter in letters:
        letter_sentences = []
        for sentence in letter[1].sents:
            letter_sentences.append(sentence)
        sentences.append([letter[0], letter_sentences])
    return sentences


# Extract all candidate terms from the input letters
def get_candidate_terms(letters):
    raw_terms = []
    clean_terms = []

    for letter in letters:
        raw_letter_terms = []
        clean_letter_terms = []

        for sentence in letter[1]:
            sentence_words = str(sentence).split()

            # max n_gram value
            sent_len = len(sentence_words)
            if sent_len > 10:
                n = 10
            else:
                n = sent_len

            rct, cct = get_candidate_terms_helper(sentence_words, n)
            raw_letter_terms += rct
            clean_letter_terms += cct

        raw_terms.append([letter[0], raw_letter_terms])
        clean_terms.append([letter[0], clean_letter_terms])

    return raw_terms, clean_terms


# Cycle through letter words to perform sliding window
def get_candidate_terms_helper(sentence_words, word_count):
    raw_terms = []
    clean_terms = []

    for j in range(word_count + 1):
        for i in range(len(sentence_words) - j + 1):
            raw_term = list_to_string(
                list(sliding_window(sentence_words[i:], j)))
            #clean_term = get_clean_term(raw_term)
            clean_term = get_alnum_term(raw_term.lower())
            final_clean_term = []
            for word in clean_term.split():
                if word not in stopwords:
                    final_clean_term.append(word)
            clean_term = ' '.join(final_clean_term)

            if raw_term not in raw_terms and clean_term not in clean_terms and raw_term != '' and clean_term != None:
                raw_terms.append(raw_term)
                clean_terms.append(clean_term)

    return raw_terms, clean_terms


# Return single-spaced string from list
def list_to_string(my_list):
    return ' '.join(str(e) for e in my_list)


def strip_front(text):
    new_text = ''
    for i in range(len(text)):
        if text[i].isalnum():
            new_text += text[i:]
            break
    return new_text


def strip_back(text):
    new_text = ''
    for i in range(len(text) - 1, 0, -1):
        if text[i].isalnum():
            new_text += text[0:i+1]
            break
    return new_text


# Remove all non-alphanumeric characters (excl. spaces and numbers) from the input term
def get_alnum_term(term):
    alnum_term = ''
    for char in term:
        if (char not in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'} and char.isalnum()) or (char == '-' or char == ' '):
            alnum_term += char
    return ' '.join(alnum_term.split())


# Get sliding window terms
def sliding_window(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        return result
    for elem in it:
        result = result[1:] + (elem,)
        return result


# Find best match of input terms between specified thresholding, using cosine similarity
def medgate_trial_json(lower, upper, clean_terms, raw_terms=None):
    matches = []

    for letter in range(len(clean_terms)):
        for term in range(len(clean_terms[letter][1])):
            raw_term = raw_terms[letter][1][term]
            clean_term = clean_terms[letter][1][term]
            stripped_term = strip_front(strip_back(raw_term)).lower()
            results = searcher.ranked_search(clean_term, lower)

            for result in results:
                if result[0] <= lower:
                    break

                if result[0] <= upper and len(result[1].split()) <= len(clean_term.split()):
                    score = result[0]
                    match = result[1].lower()

                    if score == 1 and match == stripped_term:
                        break
                    elif score == 1:
                        score = 0.9999999

                    # Change to customise JSON output (below shows DESCRIPTION: VARIABLE)
                    # Letter Name:      clean_terms[letter][0]
                    # Raw term:         raw_term
                    # Clean term:       clean_term
                    # UMLS match:       match
                    # CUI:              *removed
                    # Cosine score:     score
                    matches.append([raw_term, match])
                    break

    return matches


# Get most similar UMLS term and output to file
def main(letter, lower, upper, output_file):
    sentences = get_sentences(letters)
    raw_terms, clean_terms = get_candidate_terms(sentences)

    results = medgate_trial_json(lower, upper, clean_terms, raw_terms)

    with open(output_file, 'w+') as f:
        json.dump(results, f)


try:
    nlp = spacy.load('en_core_web_md')
except:
    os.system('python3 -m spacy download en_core_web_md')
    nlp = spacy.load('en_core_web_md')

stopwords = set(
    open(os.path.join(sys.path[0], 'stopwords.txt')).read().split('\n'))
database = load_pickle(os.path.join(sys.path[0], 'db.pickle'), 'rb')
searcher = Searcher(database, CosineMeasure())

# Input directory of letters (finds all .txt files and ignores rest)
letter_dir = os.path.join(sys.path[0], os.environ["input_dir"])
letter_type = '.txt'

# Read in letters
letters = get_letters_incl_spacy(letter_dir, letter_type)

# Cosine thresholds
lower_threshold = 0.95
upper_threshold = 1.00

# Output file name
output_dir = os.path.join(sys.path[0],os.environ["output_dir"])
os.makedirs(output_dir, exist_ok=True)
output = os.path.join(output_dir, 'TestFile.json')

main(letters, lower_threshold, upper_threshold, output)
