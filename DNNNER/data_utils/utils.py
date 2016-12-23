import re
import itertools
import codecs
import os

import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.iteritems()}


def flatten1(lst):
    return list(itertools.chain.from_iterable(lst))


def canonicalize_digits(word):
    if any([c.isalpha() for c in word]):
        return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "")  # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if wordset and (word in wordset):
            return word
        word = canonicalize_digits(word)  # try to canonicalize numbers
    if (not wordset) or (word in wordset):
        return word
    else:
        return "UUUNKKK"  # unknown token


def load_dataset(fname):
    docs = []
    with codecs.open(fname, "r", encoding="utf8") as fd:
        cur = []
        for line in fd:
            if len(line.strip()) == 0:
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else:  # read in tokens
                cur.append(line.strip().split("\t", 1))
        # flush running buffer
        docs.append(cur)
    return docs


def pad_sequence(seq, left=1, right=1):
    return left*[("<s>", "")] + seq + right*[("</s>", "")]


# For window models
def seq_to_windows(words, tags, word_to_num, tag_to_num, left=1, right=1):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>" or words[i] == "</s>":
            continue  # skip sentence delimiters
        tagn = tag_to_num[tags[i]]
        idxs = [word_to_num[words[ii]]
                for ii in range(i - left, i + right + 1)]
        X.append(idxs)
        y.append(tagn)
    return np.array(X), np.array(y)


def docs_to_windows(docs, word_to_num, tag_to_num, wsize=3):
    pad = (wsize - 1)/2
    docs = flatten1([pad_sequence(seq, left=pad, right=pad) for seq in docs])
    words, tags = zip(*docs)
    words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]
    return seq_to_windows(words, tags, word_to_num, tag_to_num, pad, pad)


def load_wt(file_path, data="", pre=False):
    if not os.path.exists(file_path + data + "vocab"):
        fp = codecs.open(file_path + data + "train", "r", encoding="utf8")
        # word_to_num = {"UUUNKKK": 0, "<s>": 1, "</s>": 2}
        word_to_num = {"UUUNKKK": 0, "<s>": 1}
        tag_to_num = {}
        word_num = 2
        tag_num = 0
        # tag dic
        for line in fp:
            line = line.strip()
            if not line:
                continue
            word_tag = line.split("\t")
            # if word_tag[0] not in word_to_num:
            #     word_to_num[word_tag[0]] = word_num
            #     word_num += 1
            if word_tag[1] not in tag_to_num:
                tag_to_num[word_tag[1]] = tag_num
                tag_num += 1
        # word vec and word dic
        word_to_vec = []
        wv_fp = codecs.open("../../data/word_embedding/seg_300.vec", "r", "utf8")
        # wv_fp = codecs.open(file_path + data + "seg_300.vec", "r", encoding="utf8")
        for line in wv_fp:
            line = line.strip()
            if not line:
                continue
            spl = line.split()
            if spl[0] in word_to_num or len(spl) % 10 == 0:
                continue
            word_to_num[spl[0]] = word_num
            word_num += 1
            vec = [float(i) for i in spl[1:]]
            word_to_vec.append(vec)
        vec_length = len(word_to_vec[0])
        bound = np.sqrt(6) / np.sqrt(word_num + vec_length)
        word_to_vec.insert(0, np.random.uniform(-bound, bound, vec_length))
        word_to_vec.insert(0, np.random.uniform(-bound, bound, vec_length))
        word_to_vec = np.array(word_to_vec)
        # num to *
        num_to_word = invert_dict(word_to_num)
        num_to_tag = invert_dict(tag_to_num)
        vocab_fp = codecs.open(file_path + data + "vocab", "w", encoding="utf8")
        # write dump file
        for k, v in num_to_word.iteritems():
            vocab_fp.write(v)
            vocab_fp.write("\n")
        vocab_fp.close()
        tag_fp = codecs.open(file_path + data + "tag", "w", encoding="utf8")
        for k, v in num_to_tag.iteritems():
            tag_fp.write(v)
            tag_fp.write("\n")
        tag_fp.close()
    else:
        fp = codecs.open(file_path + data + "vocab", "r", encoding="utf8")
        num = 0
        word_to_num = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            word_to_num[line] = num
            num += 1
        fp.close()
        num_to_word = invert_dict(word_to_num)
        fp = codecs.open(file_path + data + "tag", "r", encoding="utf8")
        num = 0
        tag_to_num = {}
        for line in fp:
            line = line.strip()
            if not line:
                continue
            tag_to_num[line] = num
            num += 1
        fp.close()
        num_to_tag = invert_dict(tag_to_num)
        # word vec
        word_to_vec = []
        if not pre:
            wv_fp = codecs.open("../../data/word_embedding/seg_300.vec", "r", "utf8")
            repeat_word = []
            for line in wv_fp:
                line = line.strip()
                if not line:
                    continue
                spl = line.split()
                if spl[0] in repeat_word or len(spl) % 10 == 0:
                    continue
                repeat_word.append(spl[0])
                vec = [float(i) for i in spl[1:]]
                word_to_vec.append(vec)
            vec_length = len(word_to_vec[0])
            bound = np.sqrt(6) / np.sqrt(len(word_to_num) + vec_length)
            word_to_vec.insert(0, np.random.uniform(-bound, bound, vec_length))
            word_to_vec.insert(0, np.random.uniform(-bound, bound, vec_length))
            word_to_vec = np.array(word_to_vec)
    return word_to_num, num_to_word, tag_to_num, num_to_tag, word_to_vec


def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if np.any(orig_y) else None
    else:
        data_X = orig_X
        data_y = orig_y
    ###
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        # Convert our target from the class index to a one hot vector
        y = None
        if np.any(data_y):
            y_indices = data_y[batch_start:batch_start + batch_size]
            y = np.zeros((len(x), label_size), dtype=np.int32)
            y[np.arange(len(y_indices)), y_indices] = 1
        ###
        yield x, y
        total_processed_examples += len(x)
