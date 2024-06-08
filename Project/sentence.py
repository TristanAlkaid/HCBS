import os
import string
import numpy as np
import torch
from gensim.models import Word2Vec

def get_sentence_vector(sentence,path):
    model = Word2Vec.load(path)
    sentence_vector = np.zeros((model.vector_size,))

    # 遍历句子中的单词并获取它们的词嵌入向量
    for word in sentence:
        if word in model.wv:
            sentence_vector += model.wv[word]

    # 计算句子向量的平均值，如果句子中没有已知单词，则结果仍然是零向量
    if np.count_nonzero(sentence_vector) > 0:
        sentence_vector /= np.count_nonzero(sentence_vector)

    sentence_vector = sentence_vector.reshape(1, 288, -1)
    # print(sentence_vector.shape)
    # 将 NumPy 数组转换为 PyTorch 张量
    sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32)

    return sentence_tensor


def save_model(sentences, path):
    model = Word2Vec(sentences, vector_size=288*288, window=10, min_count=1, sg=0)
    # model = Word2Vec(sentences, vector_size=288, window=10, min_count=1, sg=0)
    model.save(path)

def get_sentences(path="data/content_sentence"):
    text_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    translator = str.maketrans('', '', string.punctuation)
                    text = text.translate(translator)
                    text =text.lower().split()
                    text_list.append(text)
    return text_list


if __name__ == "__main__":
    sentences = get_sentences("data/content_sentence")
    save_model(sentences, "data/word2vec.model")
    path = "data/word2vec.model"
    sentence = "I am a human"
    sentence_vector = get_sentence_vector(sentence,path)
    print(sentence_vector)
