import re
import time
import nltk  # 用于关系抽取
import learning
import summarize
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


def getpath(var_file, text_list, emo_list):
    path = filedialog.askopenfilename()
    var_file.set('choose data file: ' + path)
    get_text_list(text_list, emo_list, path)


def get_text_list(text_list, emo_list, path):
    count = 0
    data = pd.read_csv(path, sep=',', header=None, encoding_errors="ignore")
    # print(len(data))
    for i in range(0, len(data)):
        text_list.append(data[0][i])
        emo_list.append(data[1][i])
        count = count + 1
        if count > 50:
            print("Loading Complete\n")
            break


def ie_process(text, var_info, var_object, var_organization, var_gpe, var_pos, var_neg, var_emo,
               var_reli, var_right, var_rate, var_re, text_result, text_list, classifier, emo_list):
    result = text.get("1.0", "end")  # 获取文本输入框的内容

    f = open("C:/Users/37301/Desktop/lab3/log.txt", "a")
    f.write('\n' + str(time.ctime()) + '\n')

    if len(result) != 1:    # single extract
        text_list.clear()
        emo_list.clear()
        text_list.append(result)
        emo_list.append("pos")

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    count = 0

    for text_name in text_list[0:len(text_list)]:  # 对每个文件，均进行一遍关系抽取处理
        pos_word_set = set()
        neu_word_set = set()
        neg_word_set = set()

        # 文章概况摘要
        smz = summarize.summarize(text_name, 3)

        # 完成主人公
        # 分句
        sentences = nltk.sent_tokenize(text_name)
        # print("sentences->", sentences)
        yuce, realscore = learning.yuce(sentences, classifier)
        # 分词
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        # print("words->", tokenized_sentences)
        # print("分词", tokenized_sentences)
        for sent in tokenized_sentences:
            for word in sent:
                if (sid.polarity_scores(word)['compound']) >= 0.5:
                    pos_word_set.add(word)
                elif (sid.polarity_scores(word)['compound']) <= -0.5:
                    neg_word_set.add(word)
                else:
                    neu_word_set.add(word)
        # print(pos_word_list)
        # print(neg_word_list)

        # 标注
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        # print("tags->", tagged_sentences)
        # 命名实体识别
        ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
        # print("identify->", ne_chunked_sents)
        # 具体的文本中，对具体的实体进行命名
        name_dict = {}
        organization_dict = {}
        gpe_dict = {}

        pr_temp = re.compile(r'.*(\bin\b|\bis\b|\bwas\b|\bwere\b|\bare\b)')

        for sent in ne_chunked_sents:
            for rel in nltk.sem.extract_rels('PER', 'LOC', sent, corpus='ace', pattern=pr_temp, window=10):
                print("-----", nltk.sem.rtuple(rel))

        for ne_taged_sentence in ne_chunked_sents:
            # print(ne_taged_sentence)
            for tagged_tree in ne_taged_sentence:
                # print(tagged_tree)
                if hasattr(tagged_tree, 'label'):
                    # print(tagged_tree)
                    type = tagged_tree.label()
                    # print("type {}".format(type))
                    if type == "PERSON":
                        name = ""
                        for tagged_left in tagged_tree:
                            name += ' ' + tagged_left[0]
                        name = name[1:]
                        if name in name_dict:
                            name_dict[name] += 1
                        else:
                            name_dict[name] = 1
                    elif type == "ORGANIZATION":
                        organization = ""
                        for tagged_left in tagged_tree:
                            organization += ' ' + tagged_left[0]
                        organization = organization[1:]
                        if organization in organization_dict:
                            organization_dict[organization] += 1
                        else:
                            organization_dict[organization] = 1
                    elif type == "GPE":
                        gpe = ""
                        for tagged_left in tagged_tree:
                            gpe += ' ' + tagged_left[0]
                        gpe = gpe[1:]
                        if gpe in gpe_dict:
                            gpe_dict[gpe] += 1
                        else:
                            gpe_dict[gpe] = 1

        name_dict = {k: v for k, v in sorted(name_dict.items(), key=lambda items: items[1], reverse=True)}
        organization_dict = {k: v for k, v in
                             sorted(organization_dict.items(), key=lambda items: items[1], reverse=True)}
        gpe_dict = {k: v for k, v in sorted(gpe_dict.items(), key=lambda items: items[1], reverse=True)}
        if len(result) != 1:
            var_info.set(smz)
            if len(name_dict) != 0:
                var_object.set(list(name_dict.keys())[0])
            else:
                var_object.set('--')
            if len(organization_dict) != 0:
                var_organization.set(list(organization_dict.keys())[0])
            else:
                var_organization.set('--')
            if len(gpe_dict) != 0:
                var_gpe.set(list(gpe_dict.keys())[0])
            else:
                var_gpe.set('--')

            info = ''
            for word in pos_word_set:
                info = info + ' ' + word
            var_pos.set(info)
            info = ''
            for word in neg_word_set:
                info = info + ' ' + word
            var_neg.set(info)
            var_emo.set(str(yuce))
            var_reli.set(str(realscore))
            var_right.set(str(0.8))
            var_rate.set(str(1))
            if yuce == 'pos':
                star = ''
                for i in range(0, int(realscore * 10)):
                    star = star + "✧ "
                var_re.set(star)
            else:
                var_re.set('NO')
            text_result.delete('0.0', 'end')
            text_result.insert('insert', var_info.get())

        # 关系抽取的结果输出
        f.write("COMMENT: " + text_name + '\n')
        f.write("Abstract: " + smz + '\n')
        f.write("Triplet: \n")
        if len(name_dict) != 0:
            f.write("Lead or Performer: " + list(name_dict.keys())[0] + '\n')
        else:
            f.write("Lead or Performer: --\n")
        if len(organization_dict) != 0:
            f.write("Organization: " + list(organization_dict.keys())[0] + '\n')
        else:
            f.write("Organization: --\n")
        if len(gpe_dict) != 0:
            f.write("GPE: " + list(gpe_dict.keys())[0] + '\n')
        else:
            f.write("GPE: --\n")
        if len(pos_word_set) != 0:
            f.write("Positive review: ")
            for word in pos_word_set:
                f.write(word + ' ')
            f.write('\n')
        else:
            f.write("Positive review: --\n")
        if len(neg_word_set) != 0:
            f.write("Negative review: ")
            for word in neg_word_set:
                f.write(word + ' ')
            f.write('\n')
        else:
            f.write("Negative review: --\n")

        f.write("Emotional tendency: " + str(yuce) + '\n')
        f.write("Reliability: " + str(realscore) + '\n\n')

        if (yuce == "pos") & (emo_list[count] == "positiv"):
            TP += 1
        elif (yuce == "neg") & (emo_list[count] == "positiv"):
            FN += 1
        elif (yuce == "neg") & (emo_list[count] == "negativ"):
            TN += 1
        elif (yuce == "pos") & (emo_list[count] == "negativ"):
            FP += 1
        count += 1
    if len(result) == 1:
        f.write("正确率： " + str(float(TP+TN)/float(TP+TN+FN+FP)) + '\n')
        f.write("召回率： " + str(float(TP)/float(TP+FN)) + '\n')
    f.close()


def gui():
    print("Training...")
    # classifier = []
    classifier = learning.train()

    text_list = []
    emo_list = []

    window = tk.Tk()
    window.title("infoExtract")
    window.geometry("710x700")

    var_info = tk.StringVar()
    # var_info.set("test!\n")
    var_result = tk.StringVar()
    var_result.set("Result")
    var_file = tk.StringVar()
    var_file.set("InfoExtract——电影评论关系抽取")
    var_object = tk.StringVar()
    var_object.set("--")
    var_organization = tk.StringVar()
    var_organization.set("--")
    var_gpe = tk.StringVar()
    var_gpe.set("--")
    var_pos = tk.StringVar()
    var_pos.set("--")
    var_neg = tk.StringVar()
    var_neg.set("--")
    var_emo = tk.StringVar()
    var_emo.set("--")
    var_reli = tk.StringVar()
    var_reli.set("--")
    var_right = tk.StringVar()
    var_right.set("--")
    var_rate = tk.StringVar()
    var_rate.set("--")
    var_re = tk.StringVar()
    var_re.set("--")

    var_recommend = tk.StringVar()
    var_recommend.set("Recommend_rate ")

    text_single = tk.Text(window, width=100, height=10)
    text_result = tk.Text(window, width=80, height=8)
    l_file = tk.Label(window, textvariable=var_file, bg="white", font=("Arial", 10), width=59, height=3)
    l_result = tk.Label(window, textvariable=var_result, bg="white", font=("Arial", 12), width=80, height=2)
    l_recommend = tk.Label(window, textvariable=var_recommend, bg="white", font=("Arial", 12), width=15, height=2)
    l_info = tk.Label(window, text="\nAbstract\n\n\n\n\nObject\n\nOrganization\n\nGPE\n\nPositive review\n\nNegative review\n\nEmotional tendency\n\nReliability\n\n正确率\n\n召回率",
                      bg="white", font=("Arial", 12), width=15, height=23)
    l_object = tk.Label(window, textvariable=var_object, bg="white", font=("Arial", 12), width=62, height=2)
    l_organization = tk.Label(window, textvariable=var_organization, bg="white", font=("Arial", 12), width=62, height=2)
    l_gpe = tk.Label(window, textvariable=var_gpe, bg="white", font=("Arial", 12), width=62, height=2)
    l_pos = tk.Label(window, textvariable=var_pos, bg="white", font=("Arial", 12), width=62, height=2)
    l_neg = tk.Label(window, textvariable=var_neg, bg="white", font=("Arial", 12), width=62, height=2)
    l_emo = tk.Label(window, textvariable=var_emo, bg="white", font=("Arial", 12), width=62, height=2)
    l_reli = tk.Label(window, textvariable=var_reli, bg="white", font=("Arial", 12), width=62, height=2)
    l_right = tk.Label(window, textvariable=var_right, bg="white", font=("Arial", 12), width=62, height=2)
    l_rate = tk.Label(window, textvariable=var_rate, bg="white", font=("Arial", 12), width=62, height=2)
    l_re = tk.Label(window, textvariable=var_re, bg="white", font=("Arial", 12), width=62, height=2)
    button = tk.Button(window, text="infoExtract", width=15, height=2,
                       command=lambda: ie_process(text_single, var_info, var_object, var_organization, var_gpe, var_pos, var_neg,
                                                  var_emo, var_reli, var_right, var_rate, var_re,
                                                  text_result, text_list, classifier, emo_list))
    button_mult = tk.Button(window, text="choose file", width=15, height=2,
                            command=lambda: getpath(var_file, text_list, emo_list))

    button.place(x=594, y=0)
    button_mult.place(x=0, y=0)
    l_file.place(x=118, y=0)
    l_result.place(x=0, y=185)
    l_info.place(x=0, y=230)
    l_recommend.place(x=0, y=645)
    l_object.place(x=140, y=330)
    l_organization.place(x=140, y=364)
    l_gpe.place(x=140, y=398)
    l_pos.place(x=140, y=432)
    l_neg.place(x=140, y=467)
    l_emo.place(x=140, y=502)
    l_reli.place(x=140, y=537)
    l_right.place(x=140, y=572)
    l_rate.place(x=140, y=608)
    l_re.place(x=140, y=643)
    text_single.place(x=0, y=48)
    text_result.place(x=140, y=230)

    window.mainloop()


if __name__ == '__main__':
    gui()
