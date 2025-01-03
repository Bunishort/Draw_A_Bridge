from context import sample
import numpy as np

nx=8
ny=8

solid = np.zeros([nx,ny])
elas_lambda = 1
elas_mu = 1
lm=1


test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm)
1+1