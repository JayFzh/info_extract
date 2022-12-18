from distance import jaccard
from operator import itemgetter
from itertools import combinations
from networkx import Graph, pagerank
from nltk import corpus, tokenize, stem


class LanguageProcessor(object):
    def __init__(self, language):
        self.language = language
        self.stopwords = corpus.stopwords.words(language)
        self.stemmer = stem.SnowballStemmer(language)

    def split_sentences(self, text):
        return tokenize.sent_tokenize(text, self.language)

    def extract_significant_words(self, sentence):
        return set(
            word for word in tokenize.word_tokenize(sentence)
            if word.isalnum() and word not in self.stopwords
        )

    def stem(self, word):
        return self.stemmer.stem(word)


def summarize(text, sentence_count=5, language='english'):
    processor = LanguageProcessor(language)

    sentence_list = processor.split_sentences(text)
    wordset_list = map(processor.extract_significant_words, sentence_list)
    stemsets = [
        {processor.stem(word) for word in wordset}
        for wordset in wordset_list
    ]

    graph = Graph()
    pairs = combinations(enumerate(stemsets), 2)
    for (index_a, stems_a), (index_b, stems_b) in pairs:
        if stems_a and stems_b:
            similarity = 1 - jaccard(stems_a, stems_b)
            if similarity > 0:
                graph.add_edge(index_a, index_b, weight=similarity)

    ranked_sentence_indexes = list(pagerank(graph).items())
    if ranked_sentence_indexes:
        sentences_by_rank = sorted(
            ranked_sentence_indexes, key=itemgetter(1), reverse=True)
        best_sentences = map(itemgetter(0), sentences_by_rank[:sentence_count])
        best_sentences_in_order = sorted(best_sentences)
    else:
        best_sentences_in_order = range(min(sentence_count, len(sentence_list)))

    return ' '.join(sentence_list[index] for index in best_sentences_in_order)
