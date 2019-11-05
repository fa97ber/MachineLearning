import numpy as np
import pandas as pd

def pca(df, r):
    cols = df.columns
    
    #Zentrieren
    means = []
    for col in cols:
        df[col] = df[col] - np.mean(df[col])
        means.append(np.mean(df[col]))
        
    #Normieren der Varianz
    for col in cols:
        df[col] = df[col] / np.std(df[col])
    
    #In Matrix umwandeln
    n, d = np.shape(df)
    X = np.zeros((n, d))
    for ni in range(n):
        for di in range(d):
            X[ni, di] = df[cols[di]][ni]
    
    #Singulärwertzerlegung
    u,d,vt = np.linalg.svd(X, full_matrices=False)
    v = vt.transpose()
    
    ud = u * d
    pc = []
    ai = []
    
    for ri in range(r):
        pc.append(v[:,ri])
        ai.append(ud[:,ri])
        
    return pc,ai,np.power(d, 2)/(n-1), means