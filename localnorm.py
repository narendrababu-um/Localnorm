#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[2]:


def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)   
    return (words, np.array(We))

def get_localnorm(We):
    G = We.mean(axis=0)
    ln = []
    for v in We:
        ln.append(distance.cityblock(G,v))
    ln = np.array(ln)
    ln = scaler.fit_transform(ln.reshape(-1, 1))
    ln = ln.flatten()
    lnweight4ind = {}
    for idx, w in enumerate(ln):
        lnweight4ind[idx] = w
    return lnweight4ind

def sim_evaluate_all(We, words, weight4ind, scoring_function, m):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["MSRpar2012",
            "MSRvid2012",
            "SMTeuro2012",
            "OnWN2012-own",
            "SMTnews2012",
            "headline2013",
            "OnWN2013",
            "FNWN2013",
            "SMT2013",
            "deft-forum2014",
            "deft-news2014",
            "headline2014",
            "images2014",
            "OnWN2014",
            "tweet-news2014",
            "answer-forum2015",
            "answer-student2015",
            "belief2015",
            "headline2015",
            "images2015",
            "sicktest",
            "twitter",
            "sts-bench"]

    for i in farr:
        p,s = sim_getCorrelation(We, words, prefix+i, weight4ind, scoring_function, m)
        parr.append(p); sarr.append(s)

    parr = np.array(parr)*100.0;sarr=np.array(sarr)*100.0
    s = ""
    counter = 0
    for i,j in zip(farr, parr):
        counter = counter+1
        s += "%30s %10f\n" % (i, j)
        if counter==5:
            n = sum(parr[0:5]) / 5.0
            s += "%30s %10f \n \n" % ("2012-average ", n)
        if counter==9:
            n = sum(parr[5:9]) / 4.0
            s += "%30s %10f \n \n" % ("2013-average ", n)
        if counter==15:
            n = sum(parr[9:15]) / 6.0
            s += "%30s %10f \n \n" % ("2014-average ", n)
        if counter==20:
            n = sum(parr[15:20]) / 5.0
            s += "%30s %10f \n \n" % ("2015-average ", n)
            
    print s

    return parr, sarr

def sim_getCorrelation(We,words,f, weight4ind, scoring_function, m):
    f = open(f,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    m1 = seq2weight(x1, m1, weight4ind)
    m2 = seq2weight(x2, m2, weight4ind)
    scores = scoring_function(We,x1,x2,m1,m2, m)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1
    
def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in xrange(seq.shape[0]):
        for j in xrange(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight

def weighted_average(We,x1,x2,w1,w2,m):
    emb1 = get_embedding(We, x1, w1, m)
    emb2 = get_embedding(We, x2, w2, m)

    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm
    return scores

def get_embedding(We, x, w, m):
    emb = get_weighted_average(We, x, w)
    if  m > 0:
        emb = Denoise_SentenceEmbeddings(emb, m)
    return emb

def Denoise_SentenceEmbeddings(X, m):
    proj = lambda a, b: a.dot(b.transpose()) * b
    svd = TruncatedSVD(n_components=m, random_state=0).fit(X)
    for i in range(m):
        lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
        pc = svd.components_[i]
        X = np.array([ v_s - lambda_i * proj(v_s, pc) for v_s in X ])
    return X

def get_weighted_average(We, x, w): 
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        nonzerolen = np.nonzero([w[i,:]])[1]
        nonzeroweights = w[i,nonzerolen]
        nonzerowordindices = x[i,nonzerolen]
        nonzerowordvecs = We[nonzerowordindices]
        nonzerowordvecs = nonzerowordvecs * (1.0 / np.linalg.norm(nonzerowordvecs, axis=0))
        emb[i,:] = nonzeroweights.dot(nonzerowordvecs) / np.count_nonzero(w[i,:])
    return emb


# In[3]:


wordfile = '../data/glove.840B.300d.m.txt'
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
(words, We) = getWordmap(wordfile)


# In[4]:


lnweight4ind = get_localnorm(We)
parr, sarr = sim_evaluate_all(We, words, lnweight4ind, weighted_average, 5)

