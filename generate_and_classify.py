import transformers
import torch

from Phyme import Phyme, rhymeUtils as ru
import itertools
import nltk 
import gensim

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# load pretrained Google vectors
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)  
words = list(word for word in model.vocab.keys())
vectors = [model[word] for word in words]
# get t-SNE embeddings
tsne = TSNE(n_components = 10, init = 'random', random_state = 10, perplexity = 100)
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

ph = Phyme()
def get_rhymes(word):
    #Get list of rhyming words with the same syllable count
    try:
        ns = ru.count_syllables(word)
        words = ph.get_perfect_rhymes(word)
        if ns in words:
            return words[ns]
        else:
            return list(itertools.chain.from_iterable(ph.get_perfect_rhymes(word).values()))
    except:
        return [word]


def is_letter(char):
    #Helper function for word parsing
    return (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z')


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

def new_sent(string,word_list):
    #Returns a string that is the new sentence after word swaps
    #Inputs: original sentence STRING and swapped word list WORD_LIST
    #This function is not used in training
    #Only once the net is fully trained do we use this to print examples
    out = ""
    word_index = 0
    building = False
    for index,char in enumerate(string):
        if is_letter(char) and not building:
            building = True
        elif (not is_letter(char)) and (not building):
            out+=char
        elif (not is_letter(char)) and building:
            building = False
            out+=word_list[word_index]
            word_index += 1
            out+=char
                
    if building:
        out+=word_list[word_index]
        word_index += 1
    
    if word_index != len(word_list):
        print("ERROR: misalignment when reforming sentence")
    
    return out


def new_sent_grad(string,word_list,perturbed):
    #Similar to new_sent
    #Efficiently generates sentences for many perturbations
    out = [""]
    word_index = 0
    # perturbed_index = 0
    building = False
    for index,char in enumerate(string):
        if is_letter(char) and not building:
            building = True
        elif (not is_letter(char)) and (not building):
            out+=char
        elif (not is_letter(char)) and building:
            building = False
            if(perturbed[0][word_index]!=0):
                out.append(out[0])
                for i in range(len(out)-1):
                    out[i] += word_list[word_index]
                    out[i] += char
                out[len(out)-1] += perturbed[1][word_index]#[perturbed_index]
                out[len(out)-1] += char
                word_index += 1
                # perturbed_index += 1
            else:
                for i in range(len(out)):
                    out[i] += word_list[word_index]
                    out[i] += char
                word_index += 1
                
    if building:
        if(perturbed[0][word_index]!=0):
            out.append(out[0])
            for i in range(len(out)-1):
                out[i] += word_list[word_index]
            out[len(out)-1] += perturbed[1][word_index]#[perturbed_index]
            word_index += 1
            # perturbed_index += 1
        else:
            for i in range(len(out)):
                out[i] += word_list[word_index]
            word_index += 1
    
    if word_index != len(word_list):
        print("ERROR: misalignment when reforming sentence")

    if len(out) != len(perturbed[2])+1:
        print("ERROR: perturb misalignment")
        
    # if perturbed_index != len(perturbed[1]):
    #     print("ERROR: misalignment when reforming sentence")
    
    return out

    
def word_swap(word):
    #Swaps single word weighted by embedding similarity
    #TODO: Must implement long_embedding above in order to get what we want here
    rhymes = get_rhymes(word)
    WE = long_embedding(word)
    weights = []
    for r in rhymes:
        RE = long_embedding(r)
        cos = np.dot(WE,RE) / (np.sqrt(np.dot(WE,WE)) * np.sqrt(np.dot(RE,RE)))
        weights.append(1+cos)
        #Potentially tweak these weights if we don't get positive results
    return random.choices(population = rhymes,weights = weights,k=1)[0]

def swap(word_list,prob_vec,p_step=.05):
    #Creates swapped word lists, weighted by embedding similarity
    #Returns: NEW is the newly swapped word list
    #Returns: PERTURBED is an array containing gradient data and coupled perturbations of the random sentence
    #Input: WORD_LIST is the original sentence in word list form (run sent2word on the sentence string)
    #Input: PROB_VEC is a list of probabilities.  Pad the end with zeros if it's length exceeds the number of words
    #Input: P_STEP is the variable for computing the discrete
    
    new = []
    perturbed = [[],[]]
    
    for index,word in enumerate(word_list):
        samp = random.random()
        if prob_vec[index]+p_step >= samp:
            new_word = word_swap(word)
            if prob_vec[index] < samp:
                new.append(word)
                perturbed[1].append(new_word)
                perturbed[0].append(1)
                perturbed[2].append(index)
            elif prob_vec[index]-p_step < samp:
                new.append(new_word)
                perturbed[1].append(word)
                perturbed[0].append(-1)
                perturbed[2].append(index)
            else:
                new.append(new_word)
                perturbed[0].append(0)
                perturbed[1].append(new_word)
            
        else:
            new.append(word)
            perturbed[0].append(0)
            perturbed[1].append(word)
        
    return new,perturbed


def score(text, tokenizer, model):
    #GPT2 score
    #Input: TEXT is a sentence in string form
    #Input: TOKENIZER, MODEL are the pretrained things
    tokenized = tokenizer(text, return_tensors="pt")
    logits = model(**tokenized).logits
    return logits.tolist()[0][1]


def evaluate(text,prob_vec,tokenizer, model, p_step = .05):
    #THE MAIN FUNCTION
    #Input: TEXT is a sentence in string form
    #Input: PROB_VEC is a numpy array of swap probabilities
    #       with length = the number of words in TEXT (i.e. len(sent2word(text)))
    #Returns: loss for these probabilities on this sentence in a single stochastic run
    #Returns: gradient at this point in a sample of dimension corresponding to a coupling
    #       that is, gradient will have many 0 entries, but will be better for SGD
    word_list = sent2word(text)
    new,perturbed = swap(word_list,prob_vec,p_step)
    sentence_list = new_sent_grad(text,new,perturbed)
    scores = [score(sent,tokenizer,model) for sent in sentence_list]
    
    loss = -1.0*scores[0]*np.linalg.norm(prob_vec)
    
    gradient = np.zeros(len(new))
    
    for i,index in enumerate(perturbed[2]):
        prob_vec[index] += p_step*perturbed[0][index]
        p_mag = np.linalg.norm(prob_vec)
        prob_vec[index] -= p_step*perturbed[0][index]
        p_loss = -1.0*scores[index]*p_mag
        
        gradient[index] = (p_loss-loss)/p_step
    
    return loss,gradient
    

def train(sentences,tokenizer, model, p_step = .05):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True)
    W_0 = np.random.rand(90,30)
    W_1 = np.random.rand(30,30)
    
    
    text = random sentence
    


#TODO: sentences = list of all sentences

# generator = transformers.pipeline("text-generation")
