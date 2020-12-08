import gensim
import numpy as np
from sklearn.decomposition import PCA
import pickle

# load pretrained Google vectors
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  

def is_letter(char):
    #Helper function for word parsing
    return (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z') or char == "'"


def sent2word(string):
    #Convert sentence to list of words
    out = []
    last = 0
    building = False
    for index,char in enumerate(string):
        if is_letter(char) and not building:
            last = index
            building = True
        elif (not is_letter(char)) and building:
            building = False
            out.append(string[last:index])
                
    if building:
        out.append(string[last:])
        
    return out


with open("sentences.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

word_set = set()

for text in content:
    word_list = sent2word(text)
    for word in word_list:
        if word in model:
            word_set.add(word)



words = list(word_set)
print(len(words))

vectors = [model[word] for word in words]
pca = PCA(n_components = 5)
pca.fit(vectors)
vectors_pca = pca.transform(vectors)
tuples_pca = [tuple(v) for v in vectors_pca]

pca_embeddings = dict(zip(words,tuples_pca))

training_data = []
inverse_embeddings = dict()

for text in content:
    word_list = sent2word(text)
    out = []
    for word in word_list:
        if word in pca_embeddings:
            out.append(pca_embeddings[word])
    if(len(out)>0):
        while(len(out)<50):
            out.append((0.0,0.0,0.0,0.0,0.0))
        if tuple(out) not in inverse_embeddings:
            training_data.append(tuple(out))
            inverse_embeddings[tuple(out)] = text
            




pickle.dump(training_data, open( "td.p", "wb" ) )
pickle.dump(pca_embeddings, open( "embedding.p", "wb" ) )
pickle.dump(inverse_embeddings, open( "inverse.p", "wb" ) )