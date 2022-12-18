import nltk.classify.util
from nltk import pos_tag
from nltk import tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier


# 获取单词的词性
def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


def features(movie, type):
    # 读取句子
    pos_features = []
    stopWords = set(stopwords.words('english'))
    for f in movie.fileids(type):
        raw_sents = movie.raw(f)
        sents = tokenize.sent_tokenize(raw_sents)
        # 分割单词,去停用词
        raw_words = [tokenize.word_tokenize(s) for s in sents]
        words_features = [w for r in raw_words for w in r if w not in stopWords and len(w) > 2 and w.isalpha()]
        # 词性标注

        wnl = WordNetLemmatizer()
        words_tag = pos_tag(words_features)  # 获取单词词性
        lemmas_sents = []
        for tag in words_tag:
            wordnet_pos = get_pos(tag[1]) or wordnet.NOUN
            words = wnl.lemmatize(tag[0], pos=wordnet_pos)
            lemmas_sents.append(words)
        pos_features.append((extract_features(lemmas_sents), type))
    return pos_features
    # 转换为小写


def train():
    pos_features = features(movie_reviews, 'pos')
    neg_features = features(movie_reviews, 'neg')

    threshold_factor = 0.8
    pos_threshold = int(threshold_factor * len(pos_features))
    neg_threshold = int(threshold_factor * len(neg_features))
    # 提取特征 800个积极文本800个消极文本构成训练集  200+200构成测试文本
    features_train = pos_features[:pos_threshold] + neg_features[:neg_threshold]
    features_test = pos_features[pos_threshold:] + neg_features[neg_threshold:]

    # 训练朴素贝叶斯分类器
    classifier = NaiveBayesClassifier.train(features_train)
    print("\n分类器的准确性:", nltk.classify.util.accuracy(classifier, features_test))
    return classifier


# 运行分类器，获得预测结果
def yuce(reviews, classifier):
    ans = ""
    score_pos = []
    score_neg = []
    for review in reviews:
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        # 打印输出
        score = round(probdist.prob(pred_sentiment), 2)

        if score > 0.7:
            if pred_sentiment == "pos":
                score_pos.append(score)
            else:
                score_neg.append(score)

    sum_pos = 0.0
    sum_neg = 0.0

    for score in score_pos:
        sum_pos += score

    for score in score_neg:
        sum_neg += score

    real_score = 0.0
    if sum_pos > sum_neg:
        ans = "pos"
        real_score = sum_pos / len(score_pos)
    else:
        ans = "neg"
        real_score = sum_neg / len(score_neg)

    return ans, real_score
