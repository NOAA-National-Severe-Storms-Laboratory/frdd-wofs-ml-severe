from scipy.stats import spearmanr
import numpy as np 

def intersection(r1, r2):
    return list(set(r1) & set(r2))

def check_size(r1, r2):
    assert len(r1) == len(r2), 'Both rankings should be the same size'

def feature_agreement(r1, r2):
    """
    Measures the fraction of common features between the 
    sets of top-k features of the two rankings. 
    
    From Krishna et al. (2022), The Disagreement Problem in 
    Explainable Machine Learning: A Practitioner’s Perspective
    
    Parameters
    ---------------
    r1, r2 : list 
        Two feature rankings of identical shape   
    """
    check_size(r1, r2)
    k = len(r1)
    
    return len(intersection(r1, r2)) / k 

def rank_agreement(r1, r2):
    """
    Stricter than feature agreement, rank agreement checks 
    that the feature order is comparable between the two rankings.
    
    From Krishna et al. (2022), The Disagreement Problem in 
    Explainable Machine Learning: A Practitioner’s Perspective
    
    Parameters
    ---------------
    r1, r2 : list 
        Two feature rankings of identical shape   
    """
    check_size(r1, r2)
    k = len(r1)
    
    return np.sum([True if x==y else False for x,y in zip(r1,r2)]) / k 

def weak_rank_agreement(r1, r2):
    """
    Check if the rank is approximately close (within one rank).
    """
    check_size(r1, r2)
    k = len(r1)
    window_size=1

    rank_agree=[]
    for i, v in enumerate(r1):
        if i == 0:
            if v in r2[i:i+window_size+1]:
                rank_agree.append(True)
            else:
                rank_agree.append(False) 
        else:
            if v in r2[i-window_size:i+window_size+1]:
                rank_agree.append(True)
            else:
                rank_agree.append(False)
                
    return np.sum(rank_agree)/k
                
                
def rank_correlation(r1, r2):
    return spearmanr(r1, r2) 