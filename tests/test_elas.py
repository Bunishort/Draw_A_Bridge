from context import sample
import numpy as np

nx=8
ny=8

solid = np.zeros([nx,ny],dtype=bool)
elas_lambda = 1
elas_mu = 1
lm=1
ux_imp=np.zeros(solid.shape)
ux_imp[1:,1:] = np.nan
uy_imp=ux_imp


test = sample.core.ElasticProblem(solid,elas_lambda,elas_mu,lm,ux_imp,uy_imp)
print("axx")
print(test.axx)
print("axy")
print(test.axy)
print("ayy")
print(test.ayy)
print("ayx")
print(test.ayx)

1+1