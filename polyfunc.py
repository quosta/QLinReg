import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
import os
import sys

import minorminer
import dimod
from neal import SimulatedAnnealingSampler
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.system import DWaveSampler
    
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


def third_order_polynomial(
    w_0 = 0,
    w_1 = 0.95,
    w_2 = 2.7,
    w_3 = 0.46,
    datapoints = 500,
    seed = 0,
    interval = 10,
    noise_mean = 0,
    noise_variance = np.sqrt(500)):
    
    np.random.seed(seed)
    X = np.linspace(-interval,interval,datapoints)
    Y = w_0 + w_1 * X + w_2 * (X ** 2) + w_3 * (X**3) + np.random.normal(noise_mean, noise_variance, datapoints)
    
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]
    polynomial_features = PolynomialFeatures(degree=3)
    X = polynomial_features.fit_transform(X)
    
    return X, Y


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
    

def SGD(X,
        Y, 
        loss = 'squared_error',
        max_iter=1000,
        penalty = None,
        tol=None):
    
    start = time()
    sgd = SGDRegressor(loss = loss, 
                       penalty = penalty,
                       max_iter = max_iter, 
                       tol = tol)
    sgd.fit(X, Y.ravel())
    end = time()

    sgd_t = end - start

    sgd_pred = sgd.predict(X)
    sgd_rmse = np.sqrt(mean_squared_error(Y,sgd_pred))
    sgd_r2 = r2_score(Y,sgd_pred)
    
    return sgd_pred, sgd_r2, sgd_rmse, sgd_t, sgd.coef_, sgd.intercept_ 
    

def get_bqm(X,
            Y,
            p_interval = (0, 3),
            p_size = 40,
            p_type = 'linear'):

    start = time()
    if p_type == 'linear':
        p = np.linspace(p_interval[0],p_interval[1],p_size)
    elif p_type == 'geometric':
        p = np.geomspace(p_interval[0],p_interval[1],p_size)
    P = np.kron(np.eye(X.shape[1]), p.T)

    part = P.T @ X.T
    Q = part @ X @ P + np.diag(-2 * part @ Y.flatten())
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    end = time()
    
    bqm_t = end - start 
    
    return bqm, P, bqm_t

    
def SA(X, Y, bqm, P, num_reads=10):
    
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


def QA(X, Y, bqm, P, num_reads=1000):
    
    start = time()
    qa = DWaveSampler(solver='Advantage_system4.1')
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


def plot(X,
         Y,
         linreg = None,
         sgd = None,
         sa = None,
         qa = None,
         filename = None):

    plt.scatter(X[:,1], Y, s=1, color='grey')
    if linreg is not None:
        plt.plot(X[:,1], linreg, color='r', label='Closed-form')
    if sgd is not None:
        plt.plot(X[:,1], sgd, color='purple', label='SGD')
    if sa is not None:
        plt.plot(X[:,1], sa, color='green', label='SA')
    if qa is not None:
        plt.plot(X[:,1], qa, color='blue', label='QA')
    plt.xlabel('${x_n}$')
    plt.ylabel('${y_n}$')
    plt.legend()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
        

def main(savedir=None):
    X, Y = third_order_polynomial()
    
    linreg_pred, linreg_r2, linreg_rmse, linreg_t, _, __ = closed_solution(X,Y)
    sgd_ols_pred, sgd_ols_r2, sgd_ols_rmse, sgd_ols_t, _, __ = SGD(X,Y)
    sgd_huber_pred, sgd_huber_r2, sgd_huber_rmse, sgd_huber_t, _, __ = SGD(X,Y, loss='huber')

    bqm, P, bqm_t = get_bqm(X,Y)
    sa_lin_pred, sa_lin_r2, sa_lin_rmse, sa_lin_t, _, __ = SA(X, Y, bqm, P)
    qa_lin_pred, qa_lin_r2, qa_lin_rmse, qa_lin_t, connection_t, embedding_t, sampling_t, unembedding_t, _, __ = QA(X, Y, bqm, P)
   
    bqm, P, bqm_t = get_bqm(X,Y,p_type='geometric',p_interval=(10e-4,3)) 
    qa_geom_pred, qa_geom_r2, qa_geom_rmse, qa_geom_t, connection_t, embedding_t, sampling_t, unembedding_t, _, __ = QA(X, Y, bqm, P)

    timings = {'QA pipeline': ['connection', 'embedding', 'sampling', 'unembedding', 'total'],
               'time': [connection_t, embedding_t, sampling_t, unembedding_t, qa_geom_t]}
    df_timings = pd.DataFrame(timings)

    data = {'solver': ['closed_form','sgd_ols','sgd_huber','sa_lin','qa_lin', 'qa_geom'],
            'r2':[linreg_r2, sgd_ols_r2, sgd_huber_r2, sa_lin_r2, qa_lin_r2, qa_geom_r2],
            'rmse':[linreg_rmse, sgd_ols_rmse, sgd_huber_rmse, sa_lin_rmse, qa_lin_rmse, qa_geom_rmse],
            'time':[linreg_t, sgd_ols_t, sgd_huber_t, sa_lin_t, qa_lin_t, qa_geom_t]}
    df = pd.DataFrame(data) 
    
    if savedir is None:
        print(df)
        print(df_timings)
        plot(X,Y,linreg_pred,qa=qa_geom_pred)
    else:
        print(df)
        print(df_timings)
        df.to_csv(os.path.join(savedir,'results.csv'), index=False)
        df_timings.to_csv(os.path.join(savedir,'timings.csv'), index=False)
        plot(X,Y,linreg_pred,qa=qa_geom_pred,filename = os.path.join(savedir,'plot.svg'))
    
    
if __name__ == '__main__':
    savedir = sys.argv[1] if len(sys.argv[1:]) > 0 else None
    if savedir:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    
    main(savedir)
