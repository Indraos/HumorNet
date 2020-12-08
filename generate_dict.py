import gensim
import numpy as np
from sklearn.manifold import TSNE
import pickle

# load pretrained Google vectors
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  
words = list(word for word in model.vocab.keys())
vectors = [model[word] for word in words]
# get t-SNE embeddings
tsne = TSNE(n_components = 10, init = 'random', random_state = 10, perplexity = 100, method='exact')
vectors_tsne = tsne.fit_transform(vectors)
tsne_embeddings = dict(zip(words,vectors_tsne))

def long_embedding(word):
    #implement full 250 long embedding here
    #return as a numpy vec
    #have it return a vector of zeros if word is not pre-trained 
    try:
        return model[word]
    except:
        return np.zeros(300)

def short_embedding(word):
    #implement PCA reduced embedding here
    #return as a numpy vec
    #have it return a vector of zeros if word is not pre-trained 
    return tsne_embeddings[word]



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
        elif (not is_letter(char)) and (not building):
            out+=char
        elif (not is_letter(char)) and building:
            building = False
            out.append(string[last:index+1])
                
    if building:
        out.append(string[last:])
        
    return out






with open("sentences.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

dic = {}

for text in content:
    word_list = sent2word(text)
    out = []
    for word in word_list:
        out.append(tuple(short_embedding(word)))
    
    dic[tuple(out)] = text
    













pickle.dump( dic, open( "save.p", "wb" ) )
