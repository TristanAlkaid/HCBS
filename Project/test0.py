from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

# 示例文本数据
documents = [
'The image shows a soccer game in progress, with two teams playing against each other on a field. The teams are wearing uniforms of different colors, which can be identified as blue and white for one team and green and yellow for the other team.',
'The names of the teams are not visible in the image. However, the colors of their uniforms are blue and white for one team and green and yellow for the other team.',
'The soccer ball is located in the middle of the field, with players from both teams running towards it.',
'The players running towards the soccer ball are from both teams, as they are trying to gain possession and control of the ball during the game.',
'The other people in the picture are also players from both teams, who are not running towards the ball at the moment. They are likely in various positions on the field, either preparing for their next move or defending their team\'s goal.'
]


# 构建词频矩阵
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# LSA算法
lsa = TruncatedSVD(n_components=5)
lsa_result = lsa.fit_transform(X)
print('LSA结果：\n', lsa_result)

# # LDA算法
# lda = LatentDirichletAllocation(n_components=5, random_state=0)
# lda_result = lda.fit_transform(X)
# print('LDA结果：\n', lda_result)