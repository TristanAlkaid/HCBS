import spacy
from sklearn.metrics.pairwise import cosine_similarity

# 加载spaCy的英语模型
nlp = spacy.load("en_core_web_md/en_core_web_md/en_core_web_md-3.7.1")

# 示例文本
documents = [
'The image shows a soccer game in progress, with two teams playing against each other on a field. The teams are wearing uniforms of different colors, which can be identified as blue and white for one team and green and yellow for the other team.',
'The names of the teams are not visible in the image. However, the colors of their uniforms are blue and white for one team and green and yellow for the other team.',
'The soccer ball is located in the middle of the field, with players from both teams running towards it.',
'The players running towards the soccer ball are from both teams, as they are trying to gain possession and control of the ball during the game.',
'The other people in the picture are also players from both teams, who are not running towards the ball at the moment. They are likely in various positions on the field, either preparing for their next move or defending their team\'s goal.'
]

text2 = "the audience is not visible"

# 目标文本
target = "temporal soccer action detection"

def score(text, target):

    # 将文本转换为spaCy文档，自动获得词向量
    doc1 = nlp(text)
    doc2 = nlp(target)

    # 获取文档的向量（平均词向量）
    vector1 = doc1.vector
    vector2 = doc2.vector

    # 计算两个向量之间的余弦相似度
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

similarity1 = score(documents[0], target)
similarity2 = score(documents[1], target)
similarity3 = score(documents[2], target)
similarity4 = score(documents[3], target)
similarity5 = score(documents[4], target)
text2_similarity = score(text2, target)

print("Cosine Similarity1:", similarity1)
print("Cosine Similarity2:", similarity2)
print("Cosine Similarity3:", similarity3)
print("Cosine Similarity4:", similarity4)
print("Cosine Similarity5:", similarity5)
print("Cosine text2_similarity:", text2_similarity)