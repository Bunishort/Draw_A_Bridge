from core import conv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interpn


class ExplicitAnimation:
    """
    :param test: ElasticProblem class object from sample.core, with is_explicit=True
    :**kwarg fy_imp : 2d matrix of imposed volumic forces in the y direction (default 0)
    :**kwarg max_iter : maximum number of Conjugate Gradient iterations, default 20
    :**kwarg max_res : maximum (non dimensional) residual convergence tolerance, default 1e-6
    kernel_type: plane_strain, plane_stress or 2daxi (only plane strain available now)
    axx: kernel _x_x for div(sigma)
    """
    def __init__(self,solid,elas_lambda,elas_mu,lm,
                 ux_imp,uy_imp, **kwargs):
        self.solid=solid
        self.elas_lambda = elas_lambda
        self.elas_mu =elas_mu
        self.lm = lm
        self.ux_imp = ux_imp
        self.uy_imp = uy_imp
        for var in ['px_bound','py_bound','fx_imp','fy_imp']:
            setattr(self,var,kwargs.get(var,np.zeros(solid.shape)))

        self.fx_imp[np.bitwise_not(self.solid)] = 0
        self.fy_imp[np.bitwise_not(self.solid)] = 0
        self.max_iter = kwargs.get('max_iter',200)
        self.max_res = kwargs.get('max_res', 1e-6)
        for var in  ['ux','uy']:
            setattr(self,var, np.zeros(solid.shape))
