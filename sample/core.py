from logging import error
import numpy as np
# from scipy.signal import convolve2d,correlate2d
# from .convolutions import addition_convolution
from cv2 import filter2D
# from torchgen.native_function_generation import self_to_out_signature
from line_profiler import profile
import numba
###Remark : pyqt6 needed only for matplotlib show, maybe not necessary for pygame !

def get_frontier(solid):
    # Calculate all the points at the frontier of the solid
    # Frontier = at least one neighbour point is not solid (including diagonals)
    # Bulk is the solid minus the frontier
    kernel = np.ones([3, 3], dtype = np.float32)
    temp = conv(solid.astype(np.float32), kernel)
    frontier = np.bitwise_and(solid, temp < 8)
    bulk = np.bitwise_and(solid, np.bitwise_not(frontier))

    return frontier, bulk

def remove_single_points(solid):
    # Remove point with no enough neighbours
    # Point needs at least 2 consecutive neighbors to be OK
    kernel = np.array([[1, 1, 0],[0, 0, 0],[0, 0, 0]], dtype = np.float32)
    temp = conv(solid, kernel)
    temp2 = temp>=2

    kernels= [np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype = np.float32), np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype = np.float32),
              np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]], dtype = np.float32), np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]], dtype = np.float32),
              np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]], dtype = np.float32), np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype = np.float32),
              np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype = np.float32)]

    for kernel in kernels:
        temp = conv(solid, kernel)
        temp2 = np.bitwise_or(temp>=2,temp2)

    solid = temp2

    return solid

def calc_normal(solid):
    # Calculate the normals to the solid boundaries on nodes
    kernelx = np.zeros([3, 3], dtype = np.float32)
    kernelx[2, 1] = -1
    kernelx[0, 1] = 1
    kernely = np.zeros([3, 3], dtype = np.float32)
    kernely[1, 2] = -1
    kernely[1, 0] = 1

    nx = conv(solid.astype(np.float32), kernelx)
    ny = conv(solid.astype(np.float32), kernely)
    n = np.sqrt(nx ** 2 + ny ** 2)
    ok = n > 0
    nx[ok] = nx[ok] / n[ok]
    ny[ok] = ny[ok] / n[ok]
    return nx, ny

def interp_stress(xq,yq,xnodes,ynodes,sxx_x,sxy_x,syy_y,sxy_y):
    from scipy.interpolate import interpn
    # x_nodes,ynodes are vectors
    lm = xnodes[1] - xnodes[0]


    x_x = xnodes + lm/2
    y_x = ynodes
    y_y = ynodes +lm/2
    x_y = xnodes

    sxx = interpn((x_x,y_x),sxx_x,(xq,yq),bounds_error=False)
    syy = interpn((x_y, y_y), syy_y, (xq, yq),bounds_error=False)
    sxy = 1/2 * (interpn((x_x, y_x), sxy_x, (xq, yq),bounds_error=False) +
                 interpn((x_y, y_y), sxy_y, (xq, yq),bounds_error=False))

    return sxx,syy,sxy

def conv22(matrix,kernel):
    #COnvolution for 2*2 kernel with specific anchor
    return filter2D(matrix,-1,kernel,anchor=(0,0))

def conv(matrix,kernel):
    #Convolution for kernels others than 2*2
    return filter2D(matrix,-1,kernel,anchor=(-1,-1))

def conv_big(matrix, kernel):
    if kernel.shape == (2,2):
        anch = (0,0)
    else:
        anch = (-1,-1)
    return filter2D(matrix,-1,kernel,anchor=anch)

class ElasticProblem:
    """
    TODO update these comments
    :param solid: Bool 2d matrix containing the position of the solid on the grid (1 if solid, 0 if not)
    :param elas_lambda
    :param elas_mu : Lamé elastic coefficients
    :param lm : pixel length
    :param ux_imp : 2d matrix of imposed displacements in the x direction. np.Nan where no displacement is imposed
    :param uy_imp : 2d matrix of imposed displacements in the y direction np.Nan where no displacement is imposed
    :**kwarg px_bound : 2d matrix of imposed stress on solid boundary in the x direction (sig.n = (px_bound,py_bound))
    :**kwarg py_bound : 2d matrix of imposed stress on solid boundary in the y direction (default 0)
    :**kwarg fx_imp : 2d matrix of imposed volumic forces in the x direction
    :**kwarg fy_imp : 2d matrix of imposed volumic forces in the y direction (default 0)
    :**kwarg max_iter : maximum number of Conjugate Gradient iterations, default 20
    :**kwarg max_res : maximum (non dimensional) residual convergence tolerance, default 1e-6
    kernel_type: plane_strain, plane_stress or 2daxi (only plane strain available now)
    axx: kernel _x_x for div(sigma)
    axy: kernel _x_y for div(sigma)
    ayy: kernel _y_y for div(sigma)
    ux: Initial guess for x displacements (2d numpy matrix, same size as solid)
    uy: Initial guess for y displacements (2d numpy matrix, same size as solid)
    exxx/eyyy/exyx/exyy : kernels for stress computation
    frontier : 2D bool matrix of frontier position
    bulk  :2d bool matrix of bulk position
    nx/ny : normals to the frontier of the solid
    """
    def __init__(self,solid,elas_lambda,elas_mu,lm,
                 ux_imp,uy_imp, **kwargs):
        self.solid=solid
        self.elas_lambda = np.float32(elas_lambda)
        self.elas_mu = np.float32(elas_mu)
        self.lm = np.float32(lm)
        self.ux_imp = np.float32(ux_imp)
        self.uy_imp = np.float32(uy_imp)
        for var in ['px_bound','py_bound','fx_imp','fy_imp']:
            setattr(self,var,kwargs.get(var,np.zeros(solid.shape, dtype=np.float32)))

        self.fx_imp[np.bitwise_not(self.solid)] = 0
        self.fy_imp[np.bitwise_not(self.solid)] = 0
        self.max_iter = np.float32(kwargs.get('max_iter',200))
        self.max_res = np.float32(kwargs.get('max_res', 1e-6))
        for var in  ['ux','uy']:
            setattr(self,var, np.zeros(solid.shape, dtype=np.float32))
        self.kernel_type = 'plane strain'
        (self.ddx1,self.ddx2,self.ddy1,self.ddy2,self.meanx,self.meany,
         self.ddxx,self.ddyy) =\
            self.def_kernel()
        self.frontier, self.bulk = get_frontier(self.solid)
        self.nx, self.ny = calc_normal(self.solid)

        self.is_uimp = np.bitwise_and(np.bitwise_not(np.isnan(self.ux_imp)),
                                      np.bitwise_not(np.isnan(self.uy_imp)))
        self.is_uimp = np.bitwise_and(self.is_uimp, self.solid)

        self.solid_not_uimp = np.float32(np.bitwise_and(np.bitwise_not(self.is_uimp), self.solid))

        self.x_frontier_edge = conv22(self.solid.astype(np.float32),self.ddx2) != 0
        self.y_frontier_edge = conv22(self.solid.astype(np.float32), self.ddy2) != 0

        self.not_solid_x_edge = conv22(self.solid.astype(np.float32), self.ddx2**2) == 0
        self.not_solid_y_edge = conv22(self.solid.astype(np.float32), self.ddy2**2) == 0

        self.isstress_x_edge = np.float32(np.bitwise_and(np.bitwise_not(self.x_frontier_edge),
                                              np.bitwise_not(self.not_solid_x_edge)))
        self.isstress_y_edge = np.float32(np.bitwise_and(np.bitwise_not(self.y_frontier_edge),
                                              np.bitwise_not(self.not_solid_y_edge)))


        tempisddx1 = conv22(self.solid.astype(np.float32), self.ddx1) != 0
        self.x_frontier_def = np.bitwise_or(tempisddx1,
                                                self.x_frontier_edge)
        tempisddy1 = conv22(self.solid.astype(np.float32), self.ddy1) != 0
        self.y_frontier_def = np.bitwise_or(tempisddy1,
                                                self.y_frontier_edge)
        ## we could define only in_corner instead of corner for best performance...
        self.corner_def = np.bitwise_and(self.y_frontier_def, self.x_frontier_def)

        self.isddx1 = conv22(self.solid.astype(np.float32), self.ddx1 ** 2) == 2
        self.isddx2 = conv22(self.solid.astype(np.float32), self.ddx2 ** 2) == 2
        self.isddy1 = conv22(self.solid.astype(np.float32), self.ddy1 ** 2) == 2
        self.isddy2 = conv22(self.solid.astype(np.float32), self.ddy2 ** 2) == 2

        self.frontier_def = np.bitwise_or(np.bitwise_not(self.isddx1),
                                             np.bitwise_not(self.isddx2))

        self.isddx1 = np.float32(self.isddx1)
        self.isddx2 = np.float32(self.isddx2)
        self.isddy1 = np.float32(self.isddy1)
        self.isddy2 = np.float32(self.isddy2)

        self.coef = - self.elas_lambda / (self.elas_lambda + 2*self.elas_mu) #Correction coef for plane strain
        # frontiers without corners :
        self.x_frontier_def_s = np.bitwise_and(self.x_frontier_def, np.bitwise_not(self.corner_def))
        self.y_frontier_def_s = np.bitwise_and(self.y_frontier_def, np.bitwise_not(self.corner_def))
        #Preconditioning options
        self.precond = kwargs.get('precond',False)
        self.precond_type = kwargs.get('precond_type','formula')
        self.precond_n = np.float32(kwargs.get('precond_n',15))
        self.precond_xx, self.precond_xy, self.precond_yy, self.precond_yx = self.def_precond()

        self.x_frontier_def = np.where(self.x_frontier_def)
        self.y_frontier_def = np.where(self.y_frontier_def)
        self.x_frontier_def_s = np.where(self.x_frontier_def_s)
        self.y_frontier_def_s = np.where(self.y_frontier_def_s)

        self.isstress_x_edge_lambda_2mu = self.isstress_x_edge * (self.elas_lambda + 2 * self.elas_mu)
        self.isstress_y_edge_lambda_2mu = self.isstress_y_edge * (self.elas_lambda + 2 * self.elas_mu)
        self.isstress_x_edge_2mu = self.isstress_x_edge *  2 * self.elas_mu
        self.isstress_y_edge_2mu = self.isstress_y_edge *  2 * self.elas_mu
        self.elas_lambda_ratio = self.elas_lambda / (self.elas_lambda + 2 * self.elas_mu)


        self.is_explicit = kwargs.get('is_explicit',False)
        if self.is_explicit:
            self.vol_mass = np.float32(kwargs.get('vol_mass', 1))
            self.dt = np.float32(kwargs.get('dt', 1))
            self.bx, self.by = self.calc_b()
            for var in ['vx', 'vy', 'sxx_x_old', 'syy_x_old', 'sxy_x_old',
                 'sxx_y_old', 'syy_y_old', 'sxy_y_old']:
                setattr(self, var, kwargs.get(var, np.zeros(self.solid.shape, dtype = np.float32)))

            self.ratio = np.float32(kwargs.get('ratio', 0.99) ) # must be between 0 and 1
            if (self.ratio >= 1) or (self.ratio <= 0):
                print('Error : Ratio must be strictly between 0 and 1')
            self.tau = np.float32(kwargs.get('tau', 1))

            self.G0 = 1 / self.ratio
            self.G1 = 1 / (1 - self.ratio)
            self.eta1 = self.tau * (self.G1 + self.G0)

            self.explicit_a = (self.G1 + self.G0) / self.eta1
            self.explicit_b = (self.G1 * self.G0) / self.eta1

            #Calculate sound speed for check
            E = self.elas_mu * (3 * self.elas_lambda + 2 * self.elas_mu) / (self.elas_lambda + self.elas_mu)
            nu = self.elas_lambda / 2 / (self.elas_lambda + self.elas_mu)
            self.c_p = np.sqrt(E / self.ratio * (1 - nu) / (self.vol_mass * (1 + nu) * (1 - 2 * nu)))
            self.c_s = np.sqrt(E / self.ratio / (2 * (1 + nu)) / self.vol_mass)
            if self.c_p * self.dt / self.lm >1:
                print('Warning : Max Sound speed * dt / lm for Compression > 1 : ' + str(self.c_p * self.dt / self.lm))
            if self.c_s * self.dt / self.lm> 1:
                print('Warning : Max Sound speed * dt / lm for Shear > 1 : ' + str(self.c_s * self.dt / self.lm))
            if self.precond:
                self.movable = np.bitwise_and(self.solid, np.bitwise_not(self.is_uimp))
                precond_norm_xx = conv_big(self.movable, np.abs(self.precond_xx))
                precond_norm_xy = conv_big(self.movable, np.abs(self.precond_xy))
                precond_norm_yy = conv_big(self.movable, np.abs(self.precond_yy))
                precond_norm_yx = conv_big(self.movable, np.abs(self.precond_yx))

                # Remove the zeros (which whould be outside the solid exclusively, so no impact)
                precond_norm_xx[precond_norm_xx == 0] = 1
                precond_norm_xy[precond_norm_xy == 0] = 1
                precond_norm_yy[precond_norm_yy == 0] = 1
                precond_norm_yx[precond_norm_yx == 0] = 1

                self.precond_norm_x = np.sqrt(precond_norm_xx**2 + precond_norm_xy**2)
                self.precond_norm_y = np.sqrt(precond_norm_yy**2 + precond_norm_yx**2)

    def def_kernel(self):
        if self.kernel_type=='plane strain':

            #Mesh cell center def matrix
            # epsilon_xx = ddx * ux / (2 lm)
            # epsilon_yy = ddy * uy / (2 lm)
            # epsilon_xy = (ddy * ux + ddx * uy) / (4 lm)
            ddx1 = np.array([[-1,], [1,]], dtype=np.float32)
            ddx2 = np.array([[0, -1], [0, 1]], dtype=np.float32)
            ddy1 = np.array([[-1,1],], dtype=np.float32)
            ddy2 = np.array([[0, 0], [-1, 1]], dtype=np.float32)

            meanx = np.array([[1,], [1,]], dtype=np.float32)
            meany = np.transpose(meanx)

            ddxx = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
            ddyy = np.transpose(ddxx)
        else:
            raise ValueError("Only \'plane strain\' is available")
        return ddx1,ddx2,ddy1,ddy2,meanx,meany,ddxx,ddyy

    def def_precond(self):

        nx = self.precond_n
        ny = self.precond_n
        x = np.arange(nx) - np.floor((nx - 1) / 2)
        y = np.arange(ny) - np.floor((ny - 1) / 2)
        gridy, gridx = np.meshgrid(y, x)
        r = (1+np.sqrt(gridx ** 2 + gridy ** 2))

        if self.precond_type == 'formula':
            precond = 1 / (r**2)
            # precond = np.exp((1-r))
            precond_xx = precond
            precond_xy = precond * gridx * gridy / ( 1 + np.sqrt(gridx**2 * gridy**2))
            precond_yy = precond
            precond_yx = precond_xy.transpose()
        elif self.precond_type == 'robust':
            precond = 1 / (r**2)
            # precond = np.exp((1-r))
            precond_xx = precond
            precond_xy = precond * 0
            precond_yy = precond
            precond_yx = precond_xy.transpose()
        elif self.precond_type == 'linear':
            precond = 1 - (r - 1) / np.floor((nx - 1) / 2)
            precond[precond<0] = 0
            # precond = np.exp((1-r))
            precond_xx = precond
            precond_xy = precond * 0
            precond_yy = precond
            precond_yx = precond_xy.transpose()
        elif self.precond_type == 'compute':
            if self.precond_n >= 7:
                #Compute a preconditioning matrix using the results for a square with 0 displacement on edges,
                #And unit displacement in the center
                solid = np.ones([nx, ny], dtype=bool)
                ux_imp = np.zeros(solid.shape)
                ux_imp[1:-1, 1:-1] = np.nan
                uy_imp = ux_imp.copy()
                ux_imp[r == 1] = 1
                uy_imp[r == 1] = 0


                test = ElasticProblem(solid, self.elas_lambda, self.elas_mu, self.lm, ux_imp, uy_imp,
                                      max_res = self.max_res, precond_type = 'formula')
                n_iter, resx, resy, res_max_convergence, convergence_hist = test.cg_loop()
                precond_xx = test.ux
                precond_xy = test.uy
                precond_yy = test.ux.transpose()
                precond_yx = test.uy.transpose()
            else:
                error('precond_n must be >= 7 for ''compute'' option')
        elif self.precond_type == 'none':
            precond = np.ones([1,1])
            # precond = np.exp((1-r))
            precond_xx = precond
            precond_xy = precond * 0
            precond_yy = precond
            precond_yx = precond * 0
        else:
            error(str(self.precond_type) + 'not a valid precond type')

        return precond_xx, precond_xy, precond_yy, precond_yx

    def cg_loop(self):
        """
        :return: ux, uy : Displacements solutions
        :return: n_iter : number of iterations
        :return: resx,resy : residual

        Solve a_u = b
        """
        n_iter = 0
        res_max_convergence = 1e9
        convergence_hist=[]

        #Displacement BC

        self.ux[self.is_uimp] =\
            self.ux_imp[self.is_uimp]
        self.uy[self.is_uimp] = \
            self.uy_imp[self.is_uimp]

        # Calculate initial residual and search direction
        bx, by = self.calc_b()
        a_u_x,a_u_y = self.calc_a_u(self.ux,self.uy)
        resx = bx-a_u_x
        resy = by-a_u_y
        dx = resx
        dy = resy
        if self.precond:
            # preconditioning d
            pdx = conv_big(dx, self.precond_xx) + conv_big(dy, self.precond_yx)
            pdy = conv_big(dy, self.precond_yy) + conv_big(dx, self.precond_xy)
            pdx[np.bitwise_or(self.is_uimp,np.bitwise_not(self.solid))] = 0
            pdy[np.bitwise_or(self.is_uimp,np.bitwise_not(self.solid))] = 0

        else:
            pdx=dx
            pdy=dy

        # Convvergence loop
        while n_iter <= self.max_iter and res_max_convergence > self.max_res:
            #update variables
            n_iter=n_iter+1

            #Calculate new displacement field
            a_d_x,a_d_y = self.calc_a_u(pdx,pdy)
            d_a_d = np.dot(dx.ravel(),a_d_x.ravel()) + np.dot(dy.ravel(),a_d_y.ravel())
            alpha = (np.dot(resx.ravel(),dx.ravel()) + np.dot(resy.ravel(),dy.ravel())) / d_a_d

            self.ux = self.ux + alpha * pdx
            self.uy = self.uy + alpha * pdy

            #Calculate new residual
            resx = resx - alpha * a_d_x
            resy = resy - alpha * a_d_y

            #Check convergence
            res_max_convergence = np.max(
                [np.max(np.abs(resx.ravel())),
                 np.max(np.abs(resy.ravel()))])
            convergence_hist.append(res_max_convergence)

            #Calculate new search direction
            beta = -((np.dot(resx.ravel(),a_d_x.ravel())
                     + np.dot(resy.ravel(),a_d_y.ravel()))
                    / d_a_d)
            dx = resx + beta*dx
            dy = resy + beta*dy

            if self.precond:
                #preconditioning
                pdx = conv_big(dx, self.precond_xx) + conv_big(dy, self.precond_yx)
                pdy = conv_big(dy, self.precond_yy) + conv_big(dx, self.precond_xy)

                # Pas besoin de mettre ces lignes la a chaque itéaration car avec isddx/y ces points
                # ne sont pas pris en compte dans le calcul de sigma. Mais évite des additions alors...
                pdx[np.bitwise_or(self.is_uimp,np.bitwise_not(self.solid))] = 0
                pdy[np.bitwise_or(self.is_uimp,np.bitwise_not(self.solid))] = 0

            else:
                pdx = dx
                pdy = dy

        return n_iter,resx,resy,res_max_convergence,convergence_hist

    def calc_a_u(self,uxt,uyt):
        #In the bulk, a_u = div(sigma)
        #On the frontier, div(sigma) is modified to take into account the boundary condition
        #WHere the displacement is imposed, a_u=0
        #Elsewhere, it is 0
        #uxt,uyt are 2d matrices of displacements

        sxx_x,sxy_x,syy_y,sxy_y = self.calc_stress(uxt,uyt)
        return self.calc_a_u_sig(sxx_x,sxy_x,syy_y,sxy_y)

    def calc_a_u_sig(self,sxx_x,sxy_x,syy_y,sxy_y ):
        # We could remove this /lm division by multiplying b by lm
        a_u_x = (conv(sxx_x,self.ddxx / self.lm) + conv(sxy_y,self.ddyy / self.lm))
        a_u_y = (conv(syy_y, self.ddyy / self.lm) + conv(sxy_x, self.ddxx) / self.lm)

        a_u_x *= self.solid_not_uimp
        a_u_y *= self.solid_not_uimp

        return a_u_x,a_u_y

    def calc_b(self):
        #In the bulk, b= fx_imp, fy_imp (volumic forces, no inertia taken into account at this stage)
        #On the frontier, b = px_bound/lm,py_bound/lm
        # /lm so that the units are the same everywhere
        # Where displacement is imposed, b=0
        #Elsewhere, b=0

        bx = -self.fx_imp
        by = -self.fy_imp
        bx[self.frontier] -= self.px_bound[self.frontier] / self.lm
        by[self.frontier] -= self.py_bound[self.frontier] / self.lm

        bx[np.bitwise_not(np.isnan(self.ux_imp))] = 0
        by[np.bitwise_not(np.isnan(self.uy_imp))] = 0

        return bx,by

    @profile
    def calc_stress(self,uxt,uyt):
        # Calculate the stress in the center of the mesh edges

        # First, unpack self.xxx variables into local variables
        ddx1 = self.ddx1
        ddx2 = self.ddx2
        ddy1 = self.ddy1
        ddy2 = self.ddy2
        lm = self.lm

        ddx1_2lm = ddx1 / 2 / lm
        ddx1_4lm = ddx1 / 4 / lm
        ddx2_2lm = ddx2 / 2 / lm
        ddx2_4lm = ddx2 / 4 / lm
        ddy1_2lm = ddy1 / 2 / lm
        ddy1_4lm = ddy1 / 4 / lm
        ddy2_2lm = ddy2 / 2 / lm
        ddy2_4lm = ddy2 / 4 / lm

        isddx1 = self.isddx1
        isddx2 = self.isddx2
        isddy1 = self.isddy1
        isddy2 = self.isddy2


        meanx_2 = self.meanx / 2
        meany_2 = self.meany / 2
        meanx_4 = meanx_2 / 2
        meany_4 = meany_2 / 2


        # First, calculate def at mesh cell centers
        duxdx2 = conv22(uxt, ddx2_2lm)
        duxdx2 *= isddx2
        duxdy2 = conv22(uxt, ddy2_4lm)
        duxdy2 *= isddy2
        duydx2 = conv22(uyt, ddx2_4lm)
        duydx2 *= isddx2
        duydy2 = conv22(uyt, ddy2_2lm)
        duydy2 *= isddy2

        exx = conv22(uxt, ddx1_2lm)
        exx *= isddx1
        exx += duxdx2
        eyy = conv22(uyt, ddy1_2lm)
        eyy *= isddy1
        eyy += duydy2
        exy = conv22(uxt, ddy1_4lm)
        exy *= isddy1
        exy += duxdy2
        eyx = conv22(uyt, ddx1_4lm)
        eyx *= isddx1
        eyx += duydx2

        # multiply by two on frontiers to compensate where isddx/isddy = 0
        exx[self.y_frontier_def] *= 2
        eyy[self.x_frontier_def] *= 2
        exy[self.x_frontier_def] *= 2
        eyx[self.y_frontier_def] *= 2

        # frontier correction to be coherent with sigma.normal = 0
        exx[self.x_frontier_def_s] = self.coef * eyy[self.x_frontier_def_s]
        eyy[self.y_frontier_def_s] = self.coef * exx[self.y_frontier_def_s]
        exy[self.y_frontier_def_s] = -eyx[self.y_frontier_def_s]
        eyx[self.x_frontier_def_s] = -exy[self.x_frontier_def_s]

        # Average + mod to have def on edges _x perpendicular to x, and _y perpendicular to y
        # Multiplied by the right factors to calculate stress directly
        exx_x = conv22(exx + (2 * self.elas_lambda_ratio) * eyy, meany_4) # merging ex and ayy for stress computation
        exx_x += duxdx2
        exy_x = conv22(2 * exy + eyx, meany_4)#exy + eyx
        exy_x += duydx2

        # duydx2 /2 necessary for exy/eyx because of epsilonxy definition
        eyy_y = conv22(eyy + (2 * self.elas_lambda_ratio) * exx, meanx_4)
        eyy_y += duydy2
        exy_y = conv22(exy + 2 * eyx , meanx_4) # exy + eyx
        exy_y += duxdy2

        ### Now calculate stress from def #######

        # Calculate stress from def
        #sxx_x is only an alias to avoid allocating memory
        sxx_x = exx_x
        syy_y = eyy_y
        sxy_x = exy_x
        sxy_y = exy_y

        # Frontier adjustments + multiplication by elastic constants
        # sxx stress is zero on x frontier, same for syy on y frontier
        sxx_x *= self.isstress_x_edge_lambda_2mu
        syy_y *= self.isstress_y_edge_lambda_2mu
        sxy_x *= self.isstress_x_edge_2mu
        sxy_y *= self.isstress_y_edge_2mu

        return sxx_x,sxy_x,syy_y,sxy_y

    @profile
    def calc_stress_explicit(self):
        #######
        # Calculating stress for a Standard Linear Solid (Zener)
        # Viscoelastic parameters G0/G1/Eta1 normalized so that the long term response
        # is the same as the elastic response


        sxx_x,sxy_x,syy_y,sxy_y = self.calc_stress(
            self.explicit_b * self.ux + self.G0 * self.vx,
            self.explicit_b * self.uy + self.G0 * self.vy)


        temp = (1 - np.exp(-self.explicit_a * self.dt)) / self.explicit_a
        sxx_x *= temp
        sxy_x *= temp
        syy_y *= temp
        sxy_y *= temp

        temp = np.exp(-self.explicit_a * self.dt)
        sxx_x += self.sxx_x_old * temp
        sxy_x += self.sxy_x_old * temp
        syy_y += self.syy_y_old * temp
        sxy_y += self.sxy_y_old * temp

        self.sxx_x_old[:] = sxx_x
        self.sxy_x_old[:] = sxy_x
        self.syy_y_old[:] = syy_y
        self.sxy_y_old[:] = sxy_y

        return sxx_x,sxy_x,syy_y,sxy_y

    @profile
    def explicit_step(self):
        #Explicit step using LeapFrog method
        sxx_x, sxy_x, syy_y, sxy_y = self.calc_stress_explicit()
        a_u_x, a_u_y = self.calc_a_u_sig(sxx_x, sxy_x, syy_y, sxy_y )

        acc_x = ( a_u_x - self.bx ) / self.vol_mass
        acc_y = ( a_u_y - self.by ) / self.vol_mass

        # if self.precond:
        #     #repartitioning the acceleration on surrounding cells, keeping the total accel constant.
        #     # ( would need adaptation if vol mass not constant)
        #     acc_x = (conv_big(acc_x / self.precond_norm_x, self.precond_xx)
        #              + conv_big(acc_y / self.precond_norm_y, self.precond_yx))
        #     acc_y = (conv_big(acc_y / self.precond_norm_y, self.precond_yy)
        #              + conv_big(acc_x / self.precond_norm_x, self.precond_xy))
        #     acc_x[np.bitwise_not(self.movable)] = 0
        #     acc_y[np.bitwise_not(self.movable)] = 0

        self.vx += acc_x * self.dt
        self.vy += acc_y * self.dt

        self.ux += self.vx * self.dt
        self.uy += self.vy * self.dt