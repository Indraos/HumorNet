import transformers
from Phyme import Phyme, rhymeUtils as ru
import itertools
import pickle
import random
import numpy as np
import tensorflow as tf
from typing import Tuple

embedded_sentences = np.array(pickle.load(open("td.p","rb")))
pca_embeddings = pickle.load(open("embedding.p","rb" ) )
embedded_sentence_inverter = pickle.load(open( "inverse.p", "rb" ))
# embedded_sentence_inverter = tf.lookup.StaticHashTable(
#     tf.lookup.KeyValueTensorInitializer(np.array(list(inverter.keys())), np.array(list(inverter.values()))), "")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True)

def embedding(word: str) -> Tuple[int, ...]:
    """
    Retrieve embedding vectors
    """
    if word in pca_embeddings:
        return pca_embeddings[word]
    else:
        return (0.0,0.0,0.0,0.0,0.0)
    

ph = Phyme()
def get_rhymes(word: str) -> [str]:
    """
    Get list of rhyming words with the same syllable count
    """
    try:
        ns = ru.count_syllables(word)
        words = ph.get_perfect_rhymes(word)
        if ns in words:
            return words[ns]
        else:
            return list(itertools.chain.from_iterable(ph.get_perfect_rhymes(word).values()))
    except:
        return [word]


def is_letter(char: chr) -> bool:
    """
    Helper function for word parsing
    """
    return (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z') or char == "'"


def sent2word(sentence: str) -> [str]:
    """
    Convert sentence to list of words
    """
    out = []
    last = 0
    building = False
    for index,char in enumerate(sentence):
        if is_letter(char) and not building:
            last = index
            building = True
        elif (not is_letter(char)) and building:
            building = False
            out.append(sentence[last:index])    
    if building:
        out.append(sentence[last:]) 
    return out

def new_sent(string: str, word_list: [str]) -> str:
    """
    Returns a string that is the new sentence after word swaps

    This function is not used in training
    Only once the net is fully trained do we use this to print examples
    """
    out = ""
    word_index = 0
    building = False
    for _ ,char in enumerate(string):
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

def new_sent_grad(string:str,word_list:[str],perturbed:[str])->str:
    #Similar to new_sent
    #Efficiently generates sentences for many perturbations
    out = [""]
    word_index = 0
    building = False
    for _,char in enumerate(string):
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
                out[len(out)-1] += perturbed[1][word_index]
                out[len(out)-1] += char
                word_index += 1
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
            out[len(out)-1] += perturbed[1][word_index]
            word_index += 1
        else:
            for i in range(len(out)):
                out[i] += word_list[word_index]
            word_index += 1
    
    if word_index != len(word_list):
        print("ERROR: misalignment when reforming sentence")

    if len(out) != len(perturbed[2])+1:
        print("ERROR: perturb misalignment")
    return out

    
def word_swap(word: str) -> str:
    """
    Swaps single word weighted by embedding similarity
    Must implement embedding above in order to get what we want here
    """
    rhymes = get_rhymes(word)
    WE = embedding(word)
    weights = []
    for r in rhymes:
        RE = embedding(r)
        cos = np.dot(WE,RE) / (np.sqrt(np.dot(WE,WE)) * np.sqrt(np.dot(RE,RE)))
        weights.append(1+cos)
    return random.choices(population = rhymes,weights = weights,k=1)[0]

def swap(word_list:str,prob_vec: [str],p_step:float=.05) -> (str,str):
    """Creates swapped word lists, weighted by embedding similarity
    
    Input: WORD_LIST is the original sentence in word list form (run sent2word on the sentence string)
    Input: PROB_VEC is a list of probabilities.  Pad the end with zeros if it's length exceeds the number of words
    Input: P_STEP is the variable for computing the discrete
    
    Returns: NEW is the newly swapped word list
    Returns: PERTURBED is an array containing gradient data and coupled perturbations of the random sentence
    """
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


def score(text:str, tokenizer:transformers.AutoTokenizer, model:transformers.AutoModelForSequenceClassification) -> float:
    """
    GPT2 score

    Input: TEXT is a sentence in string form
    Input: TOKENIZER, MODEL are the pretrained things
    """
    tokenized = tokenizer(text, return_tensors="pt")
    logits = model(**tokenized).logits
    return logits.tolist()[0][1]

def custom_loss(input):
    @tf.custom_gradient
    def loss(prob_vec, y_pred=0):
        prob_vec = y_pred
        text = embedded_sentence_inverter[input.ref()]
        word_list = sent2word(text)
        new,perturbed = swap(word_list,prob_vec)
        sentence_list = new_sent_grad(text,new,perturbed)
        scores = [score(sent,tokenizer,model) for sent in sentence_list]
        
        loss = -1.0*scores[0]*np.linalg.norm(prob_vec)
        def grad(y, p_step=.05):
            gradient = np.zeros(len(new))
            for _,index in enumerate(perturbed[2]):
                prob_vec[index] += p_step*perturbed[0][index]
                p_mag = np.linalg.norm(prob_vec)
                prob_vec[index] -= p_step*perturbed[0][index]
                p_loss = -1.0*scores[index]*p_mag
                gradient[index] = (p_loss-loss)/p_step
        return loss, grad
    return loss

i = tf.keras.Input(shape=(50,5,))
x = tf.keras.layers.Dense(10, activation="relu")(i)
x = tf.keras.layers.Dense(5, activation="relu")(x)
x = tf.keras.layers.Lambda(lambda x: x/2)(x)
x = tf.keras.layers.Dropout(.5)(x)
model = tf.keras.Model(i,x)
model.compile(loss=custom_loss(i), optimizer=tf.optimizers.Adam(learning_rate=0.001) )
model.fit(embedded_sentences, embedded_sentences, shuffle=True, verbose=1)