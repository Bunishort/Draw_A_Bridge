import numpy as np
# from scipy.signal import convolve2d,correlate2d
# from .convolutions import addition_convolution
from cv2 import filter2D
from torchgen.native_function_generation import self_to_out_signature


###Remark : pyqt6 needed only for matplotlib show, maybe not necessary for pygame !

def get_frontier(solid):
    # Calculate all the points at the frontier of the solid
    # Frontier = at least one neighbour point is not solid (including diagonals)
    # Bulk is the solid minus the frontier
    kernel = np.ones([3, 3])
    temp = conv(solid.astype(int), kernel)
    frontier = np.bitwise_and(solid, temp < 8)
    bulk = np.bitwise_and(solid, np.bitwise_not(frontier))

    return frontier, bulk

def remove_single_points(solid):
    # Remove point with no enough neighbours
    # Point needs at least 2 consecutive neighbors to be OK
    kernel = np.array([[1, 1, 0],[0, 0, 0],[0, 0, 0]])
    temp = conv(solid, kernel)
    temp2 = temp>=2

    kernels= [np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
              np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]]), np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]]),
              np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]), np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
              np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])]

    for kernel in kernels:
        temp = conv(solid, kernel)
        temp2 = np.bitwise_or(temp>=2,temp2)

    solid = temp2

    return solid

def calc_normal(solid):
    # Calculate the normals to the solid boundaries on nodes
    kernelx = np.zeros([3, 3])
    kernelx[2, 1] = -1
    kernelx[0, 1] = 1
    kernely = np.zeros([3, 3])
    kernely[1, 2] = -1
    kernely[1, 0] = 1

    nx = conv(solid.astype(int), kernelx)
    ny = conv(solid.astype(int), kernely)
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

def conv(matrix,kernel):
    #return convolve2d(matrix,kernel,'same')
    #return correlate2d(matrix, kernel, 'same')
    #return addition_convolution(matrix,kernel)
    if kernel.shape == (2,2):
        anch = (0,0)
    else:
        anch = (-1,-1)
    return filter2D(matrix.astype(np.float32),-1,kernel.astype(np.float32),anchor=anch)


class ElasticProblem:
    """
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
        self.kernel_type = 'plane strain'
        (self.ddx1,self.ddx2,self.ddy1,self.ddy2,self.meanx,self.meany,
         self.ddxx,self.ddyy) =\
            self.def_kernel()
        self.frontier, self.bulk = get_frontier(self.solid)
        self.nx, self.ny = calc_normal(self.solid)

        self.is_uimp = np.bitwise_and(np.bitwise_not(np.isnan(self.ux_imp)),
                                      np.bitwise_not(np.isnan(self.uy_imp)))
        self.is_uimp = np.bitwise_and(self.is_uimp, self.solid)

        self.x_frontier_edge = conv(self.solid.astype(int),self.ddx2) != 0
        self.y_frontier_edge = conv(self.solid.astype(int), self.ddy2) != 0

        self.not_solid_x_edge = conv(self.solid.astype(int), self.ddx2**2) == 0
        self.not_solid_y_edge = conv(self.solid.astype(int), self.ddy2**2) == 0

        self.isstress_x_edge = np.bitwise_and(np.bitwise_not(self.x_frontier_edge),
                                              np.bitwise_not(self.not_solid_x_edge))
        self.isstress_y_edge = np.bitwise_and(np.bitwise_not(self.y_frontier_edge),
                                              np.bitwise_not(self.not_solid_y_edge))


        tempisddx1 = conv(self.solid.astype(int), self.ddx1) != 0
        self.x_frontier_def = np.bitwise_or(tempisddx1,
                                                self.x_frontier_edge)
        tempisddy1 = conv(self.solid.astype(int), self.ddy1) != 0
        self.y_frontier_def = np.bitwise_or(tempisddy1,
                                                self.y_frontier_edge)
        ## we could define only in_corner instead of corner for best performance...
        self.corner_def = np.bitwise_and(self.y_frontier_def, self.x_frontier_def)

        self.isddx1 = conv(self.solid.astype(int), self.ddx1 ** 2) == 2
        self.isddx2 = conv(self.solid.astype(int), self.ddx2 ** 2) == 2
        self.isddy1 = conv(self.solid.astype(int), self.ddy1 ** 2) == 2
        self.isddy2 = conv(self.solid.astype(int), self.ddy2 ** 2) == 2

        self.frontier_def = np.bitwise_or(np.bitwise_not(self.isddx1),
                                             np.bitwise_not(self.isddx2))
        self.coef = - self.elas_lambda / (self.elas_lambda + 2*self.elas_mu) #Correction coef for plane strain
        # frontiers without corners :
        self.x_frontier_def_s = np.bitwise_and(self.x_frontier_def, np.bitwise_not(self.corner_def))
        self.y_frontier_def_s = np.bitwise_and(self.y_frontier_def, np.bitwise_not(self.corner_def))

        self.is_explicit = kwargs.get('is_explicit',False)
        if self.is_explicit:
            self.vol_mass = kwargs.get('vol_mass', 1)
            self.dt = kwargs.get('dt', 1)
            for var in ['vx', 'vy', 'exx_x_old', 'eyy_x_old', 'exy_x_old', 'sxx_x_old', 'syy_x_old', 'sxy_x_old',
                'exx_y_old','eyy_y_old', 'exy_y_old', 'sxx_y_old', 'syy_y_old', 'sxy_y_old']:
                setattr(self, var, kwargs.get(var, np.zeros(self.solid.shape)))

            self.ratio = kwargs.get('ratio', 0.99)  # must be between 0 and 1
            if (self.ratio >= 1) or (self.ratio <= 0):
                print('Error : Ratio must be strictly between 0 and 1')
            self.tau = kwargs.get('tau', 1)

            self.G0 = 1 / self.ratio
            self.G1 = 1 / (1 - self.ratio)
            self.eta1 = self.tau * (self.G1 + self.G0)

            self.denom = 1 + self.G1 / self.G0 + self.eta1 / self.G0 / self.dt
            self.etasdt = self.eta1 / self.dt



    def def_kernel(self):
        if self.kernel_type=='plane strain':

            #Mesh cell center def matrix
            # epsilon_xx = ddx * ux / (2 lm)
            # epsilon_yy = ddy * uy / (2 lm)
            # epsilon_xy = (ddy * ux + ddx * uy) / (4 lm)
            ddx1 = np.array([[-1, 0], [1, 0]])
            ddx2 = np.array([[0, -1], [0, 1]])
            ddy1 = np.array([[-1,1],[0,0]])
            ddy2 = np.array([[0, 0], [-1, 1]])

            meanx = np.array([[1, 0], [1, 0]])
            meany = np.transpose(meanx)

            ddxx = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]])
            ddyy = np.transpose(ddxx)
        else:
            raise ValueError("Only \'plane strain\' is available")
        return ddx1,ddx2,ddy1,ddy2,meanx,meany,ddxx,ddyy

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

        # Convvergence loop
        while n_iter <= self.max_iter and res_max_convergence > self.max_res:
            #update variables
            n_iter=n_iter+1

            #Calculate new displacement field
            a_d_x,a_d_y = self.calc_a_u(dx,dy)
            d_a_d = np.dot(dx.ravel(),a_d_x.ravel()) + np.dot(dy.ravel(),a_d_y.ravel())
            alpha = (np.dot(resx.ravel(),dx.ravel()) + np.dot(resy.ravel(),dy.ravel())) / d_a_d

            self.ux = self.ux + alpha * dx
            self.uy = self.uy + alpha * dy

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
        return n_iter,resx,resy,res_max_convergence,convergence_hist

    def calc_a_u(self,uxt,uyt):
        #In the bulk, a_u = div(sigma)
        #On the frontier, div(sigma) is modified to take into account the boundary condition
        #WHere the displacement is imposed, a_u=0
        #Elsewhere, it is 0
        #uxt,uyt are 2d matrices of displacements

        sxx_x,sxy_x,syy_y,sxy_y = self.calc_stress(uxt,uyt)

        # We could remove this /lm division by multiplying b by lm
        a_u_x = (conv(sxx_x,self.ddxx / self.lm) + conv(sxy_y,self.ddyy / self.lm))
        a_u_y = (conv(syy_y, self.ddyy / self.lm) + conv(sxy_x, self.ddxx) / self.lm)

        a_u_x *= self.solid
        a_u_y *= self.solid

        a_u_x *= np.bitwise_not(self.is_uimp)
        a_u_y *= np.bitwise_not(self.is_uimp)

        return -a_u_x,-a_u_y

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

        return -bx,-by

    def calc_def(self,uxt,uyt):
        #Calculate the stress in the center of the mesh edges

        #First, calculate def at mesh cell centers
        duxdx2 = conv(uxt, self.ddx2 / (2 * self.lm))*self.isddx2
        duxdy2 = conv(uxt, self.ddy2 / (4 * self.lm))*self.isddy2
        duydx2 = conv(uyt, self.ddx2 / (4 * self.lm))*self.isddx2
        duydy2 = conv(uyt, self.ddy2 / (2 * self.lm)) * self.isddy2
        exx = conv(uxt, self.ddx1 / (2 * self.lm))*self.isddx1  + duxdx2
        eyy = conv(uyt, self.ddy1 / (2 * self.lm))*self.isddy1 + duydy2
        exy = conv(uxt, self.ddy1 / (4 * self.lm))*self.isddy1 + duxdy2
        eyx = conv(uyt, self.ddx1 / (4 * self.lm))*self.isddx1 + duydx2

        # multiply by two on frontiers to compensate where isddx/isddy = 0
        exx[self.y_frontier_def] *= 2
        eyy[self.x_frontier_def] *= 2
        exy[self.x_frontier_def] *= 2
        eyx[self.y_frontier_def] *= 2

        # frontier correction
        exx[self.x_frontier_def_s] = self.coef * eyy[self.x_frontier_def_s]
        eyy[self.y_frontier_def_s] = self.coef * exx[self.y_frontier_def_s]
        exy[self.y_frontier_def_s] = -eyx[self.y_frontier_def_s]
        eyx[self.x_frontier_def_s] = -exy[self.x_frontier_def_s]

        #Average + mod to have def on edges _x perpendicular to x, and _y perpendicular to y
        exx_x = conv(exx, self.meany / 4) + duxdx2
        eyy_x = conv(eyy, self.meany / 2)
        exy_x = conv(exy, self.meany / 2)
        eyx_x = conv(eyx, self.meany / 4) + duydx2

        #duydx2 /2 necessary for exy/eyx because of epsilonxy definition
        exx_y = conv(exx, self.meanx / 2)
        eyy_y = conv(eyy, self.meanx / 4) + duydy2
        exy_y = conv(exy, self.meanx / 4) + duxdy2
        eyx_y = conv(eyx, self.meanx / 2)

        #Calculate complete shear deformation exy
        exy_x += eyx_x
        exy_y += eyx_y

        # Adjust defs on frontier : not necessary because more efficiently done directly on stress
        # exx_x[self.x_frontier_edge] = self.coef * eyy_x[self.x_frontier_edge]
        # exy_x[self.x_frontier_edge] = 0
        # eyy_y[self.y_frontier_edge] = self.coef * exx_y[self.y_frontier_edge]
        # exy_y[self.y_frontier_edge] = 0

        return exx_x,eyy_x,exy_x, eyy_y,exx_y,exy_y

    def calc_stress_eps(self, exx_x, eyy_x, exy_x, eyy_y, exx_y, exy_y):
        #Calculate stress from def
        sxx_x = (self.elas_lambda + 2 * self.elas_mu) * exx_x + self.elas_lambda * eyy_x
        sxy_x =  (2 * self.elas_mu) * exy_x

        syy_y = (self.elas_lambda + 2 * self.elas_mu) * eyy_y + self.elas_lambda * exx_y
        sxy_y =  (2 * self.elas_mu) * exy_y

        #Frontier adjustments
        # sxx stress is zero on x frontier, same for syy on y frontier
        sxx_x *= self.isstress_x_edge
        syy_y *= self.isstress_y_edge
        sxy_x *= self.isstress_x_edge
        sxy_y *= self.isstress_y_edge

        return sxx_x,sxy_x,syy_y,sxy_y

    def calc_stress(self,uxt,uyt):
        exx_x, eyy_x, exy_x, eyy_y, exx_y, exy_y = self.calc_def(uxt,uyt)
        return self.calc_stress_eps(exx_x, eyy_x, exy_x, eyy_y, exx_y, exy_y )

    def calc_stress_explicit(self, uxt, uyt):
        #######
        # eps = self.calc_def(uxt,uyt)
        # eps_visc = kk * eps + kk * eps_old
        # sigma_temp = kk * self.calc_stress_eps(eps_visc)
        # sigma = kk * sigma_temp
        #

        def_list = np.array([
         self.exx_x_old.copy(),
         self.eyy_x_old.copy(),
         self.exy_x_old.copy(),
         self.eyy_y_old.copy(),
         self.exx_y_old.copy(),
         self.exy_y_old.copy()])

        # Could be optimzed by keeping np array in memory directly, but harder to read
        [self.exx_x_old, self.eyy_x_old, self.exy_x_old,
         self.eyy_y_old, self.exx_y_old, self.exy_y_old] = self.calc_def(uxt, uyt)
        def_list *= -(self.etasdt) / self.denom
        def_list += (self.G1 + self.etasdt) / self.denom * np.array(
            [self.exx_x_old, self.eyy_x_old, self.exy_x_old,self.eyy_y_old, self.exx_y_old, self.exy_y_old])

        sxx_x,sxy_x,syy_y,sxy_y = self.calc_stress_eps(
            *[def_list[i,:,:] for i in range(0,6)])

        sxx_x += self.etasdt / self.G0 / self.denom * self.sxx_x_old
        sxy_x += self.etasdt / self.G0 / self.denom * self.sxy_x_old
        syy_y += self.etasdt / self.G0 / self.denom * self.syy_y_old
        sxy_y += self.etasdt / self.G0 / self.denom * self.sxy_y_old

        return sxx_x,sxy_x,syy_y,sxy_y
