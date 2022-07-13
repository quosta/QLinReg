import pandas as pd
import numpy as np

from time import time
import os
import sys

import minorminer
import dimod
from neal import SimulatedAnnealingSampler
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.system import DWaveSampler
    
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_random_state

from scipy.linalg import block_diag


def closed_solution(X,Y):
    
    start = time()
    linreg = LinearRegression()
    linreg.fit(X, Y)
    end = time()
    linreg_t = end - start
    
    linreg_pred = linreg.predict(X)
    linreg_rmse = np.sqrt(mean_squared_error(Y,linreg_pred))
    linreg_r2 = r2_score(Y,linreg_pred)
    
    return linreg_pred, linreg_r2, linreg_rmse, linreg_t, linreg.coef_, linreg.intercept_ 
    

def SGD(X,Y):
    
    start = time()
    sgd = SGDRegressor()
    sgd.fit(X, Y.ravel())
    end = time()

    sgd_t = end - start

    sgd_pred = sgd.predict(X)
    sgd_rmse = np.sqrt(mean_squared_error(Y,sgd_pred))
    sgd_r2 = r2_score(Y,sgd_pred)
    
    return sgd_pred, sgd_r2, sgd_rmse, sgd_t, sgd.coef_, sgd.intercept_ 
    

def regression_dataset(n_samples = 524288,
                       n_features = 88,
                       n_informative = 88,
                       n_targets = 1,
                       noise=0.0,
                       random_state = 2):
    
    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    X = generator.randn(n_samples, n_features)
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = generator.rand(n_informative, n_targets)
    y = np.dot(X, ground_truth)
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)
    y = np.squeeze(y)
    coef = np.squeeze(ground_truth)
    
    return X, y, coef

	
def get_P(w,rate,vec_len=2):
        if vec_len == 2:
            return block_diag(*[np.array([weight-rate , weight+rate]) for weight in w])
        elif vec_len == 4:
            return block_diag(*[np.array([weight - 2*rate, weight-rate , weight+rate, weight + 2*rate]) for weight in w])


def get_bqm(X,
            Y,
            vec_len=2,
            p_interval = (1/3,2/3),
            p_type = 'fixed',
            w = None,
            rate = 0.1):

    start = time()
    
    if p_type == 'fixed':
        p = np.array([1/3,2/3])
        P = np.kron(np.eye(X.shape[1]), p.T)
    elif p_type == 'adaptive':
        if w is None:
            w = np.array([0.5 for i in range(X.shape[1])])
        P = get_P(w,rate,vec_len=vec_len)

    path = np.einsum_path('ji,kj,kl,lm -> im',P,X,X,P, optimize='optimal')[0]  
    path2 = np.einsum_path('ji,kj,k', P,X,Y, optimize='optimal')[0]
    Q = np.einsum('ji,kj,kl,lm -> im',P,X,X,P,optimize=path) + np.diag(-2 * np.einsum('ji,kj,k', P,X,Y,optimize=path2))

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    end = time()
    bqm_t = end - start 
    
    return bqm, P, bqm_t


def fixed_sa(X, Y, bqm, P, num_reads=10):    
    start = time()
    sa = SimulatedAnnealingSampler()
    sol = sa.sample(bqm,num_reads=num_reads)
    w = P @ list(sol.first.sample.values())
    end = time()
    sa_t = end - start
    
    regr = LinearRegression()
    regr.coef_ = np.array([0] + list(w[1:]))
    regr.intercept_ = w[0]
    sa_pred = regr.predict(X)
    sa_rmse = np.sqrt(mean_squared_error(Y,sa_pred))
    sa_r2 = r2_score(Y,sa_pred)
        
    return sa_pred, sa_r2, sa_rmse, sa_t, regr.coef_, regr.intercept_  


def fixed_qa(X, Y, bqm, P, num_reads=1000, qa_sampler=None):
    start = time()
    if qa_sampler == None:
        qa = DWaveSampler(solver='Advantage_system4.1')
    else:
        qa = qa_sampler
    end = time()
    connection_t = end - start
    
    start = time()
    emb = minorminer.busclique.find_clique_embedding(dimod.to_networkx_graph(bqm), qa.to_networkx_graph())
    new_bqm = embed_bqm(bqm, emb, qa.to_networkx_graph())#,chain_strength=uniform_torque_compensation)
    end = time()
    embedding_t = end - start

    start = time()
    qa_sol = qa.sample(new_bqm,num_reads=num_reads)
    end = time()
    sampling_t = end - start

    start = time()
    sol = unembed_sampleset(qa_sol, emb, bqm)
    end = time()
    unembedding_t = end - start

    w = P @ list(sol.first.sample.values())
    
    regr = LinearRegression()
    regr.coef_ = np.array([0] + list(w[1:]))
    regr.intercept_ = w[0]
    qa_pred = regr.predict(X)
    qa_rmse = np.sqrt(mean_squared_error(Y,qa_pred))
    qa_r2 = r2_score(Y,qa_pred)
    
    qa_t = connection_t + embedding_t + sampling_t + unembedding_t        
    return qa_pred, qa_r2, qa_rmse, qa_t, connection_t, embedding_t, sampling_t, unembedding_t, regr.coef_, regr.intercept_  


def adaptive(X, 
             Y, 
             solver = 'sa',
             vec_len = 2,
             num_reads = 10,
             num_iterations = 20,
             early_stop = 3,
             rate = 0.1,
             rate_descent_factor = 1.5,
             rate_ascend_factor = 1.1,
             verbose = True):
    
    start = time()
    w = None
    flag = 0
    best_r2 = -10e15
    prev_r2 = -10e15
    qa_sampler = DWaveSampler(solver='Advantage_system4.1') if solver == 'qa' else None

    for i in range(num_iterations):
        bqm, P, _ = get_bqm(X, Y, p_type='adaptive', w = w, rate = rate, vec_len = vec_len)
        if solver == 'sa':
            pred, r2, rmse, t, coef_, intercept_ = fixed_sa(X, Y, bqm, P, num_reads=num_reads)
        elif solver == 'qa':
            pred, r2, rmse, t, connection_t, embedding_t, sampling_t, unembedding_t, coef_, intercept_ = fixed_qa(X, Y, bqm, P, num_reads=num_reads, qa_sampler = qa_sampler)
        if verbose:
            print(f"iteration: {i}, current r2: {r2}, best r2: {best_r2}")
        if r2 > prev_r2:
            rate /= rate_descent_factor
            coef_[0] = intercept_
            w = coef_ if w is None else (w+coef_)/2
            flag = 0
        else:
            rate *= rate_ascend_factor
            flag += 1
    
        if r2 > best_r2: 
            best_r2 = r2
            best_rmse = rmse
            best_iter = i
            best_pred = pred
      
        prev_r2 = r2
        
        if flag == early_stop:
            if verbose:
                print("No further improvement, stopping early")
                print(f"Best r2: {best_r2}")                
            break
    end = time()
    t = end - start
    return best_pred, best_r2, best_rmse, best_iter, t, w


def main(savedir=None):
    df = pd.DataFrame() 
    vec_len = 2
    rate = 0.2
    for f in range(5,88,5):
        X, Y, coef = regression_dataset(n_samples=1000000,noise=0.3,n_features=f)
        bqm, P, bqm_t = get_bqm(X,Y,vec_len=vec_len)
        print(bqm_t)        
        linreg_pred, linreg_r2, linreg_rmse, linreg_t, _, __ = closed_solution(X,Y)
        sgd_pred, sgd_r2, sgd_rmse, sgd_t, _, __ = SGD(X,Y)
        bqm, P, bqm_t = get_bqm(X,Y,vec_len=vec_len)
        sa_pred, sa_r2, sa_rmse, sa_t, _, __ = fixed_sa(X, Y, bqm, P,num_reads=1000)
        qa_pred, qa_r2, qa_rmse, qa_t, connection_t, embedding_t, sampling_t, unembedding_t, _, __ = fixed_qa(X, Y, bqm, P)
        qa_ada_pred, qa_ada_r2, qa_ada_rmse, qa_ada_iter, qa_ada_t, _ = adaptive(X, Y, solver='qa', rate=rate,vec_len=vec_len, num_reads=1000, early_stop=5,verbose=False)
        sa_ada_pred, sa_ada_r2, sa_ada_rmse, sa_ada_iter, sa_ada_t, _ = adaptive(X, Y,solver='sa',rate = rate,vec_len=vec_len,num_iterations=1000, early_stop=5,verbose=False)
  
        data = {'features':f,
                'linreg_r2':linreg_r2,
                'sgd_r2':sgd_r2,
                'sa_r2':sa_r2,
                'qa_r2':qa_r2,
                'sa_ada_r2':sa_ada_r2,
                'qa_ada_r2':qa_ada_r2,
                'linreg_rmse':linreg_rmse,
                'sgd_rmse':sgd_rmse,
                'sa_rmse':sa_rmse,
                'qa_rmse':qa_rmse,
                'sa_ada_rmse':sa_ada_rmse,
                'qa_ada_rmse':qa_ada_rmse,
                'linreg_t':linreg_t,
                'sgd_t':sgd_t,
                'sa_t':sa_t,
                'qa_t':qa_t,
                'sa_ada_t':sa_ada_t,
                'qa_ada_t':qa_ada_t}
        df = df.append(data,ignore_index=True) 
        if savedir is None:
            print(df)
        else:
            df.to_csv(os.path.join(savedir,'results.csv'), index=False)
        
  
if __name__ == '__main__':
    savedir = sys.argv[1] if len(sys.argv[1:]) > 0 else None
    if savedir:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    
    main(savedir)
