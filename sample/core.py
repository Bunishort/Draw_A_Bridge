import numpy as np
from scipy.signal import convolve2d,correlate2d
from .convolutions import addition_convolution
from cv2 import filter2D
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

        tempisddx1 = conv(self.solid.astype(int), self.ddx1) != 0
        temp_x_frontier_def = np.bitwise_or(tempisddx1,
                                             self.x_frontier_edge)
        self.x_frontier_def = np.bitwise_and(tempisddx1,
                                                self.x_frontier_edge)
        tempisddy1 = conv(self.solid.astype(int), self.ddy1) != 0
        temp_y_frontier_def = np.bitwise_or(tempisddy1,
                                             self.y_frontier_edge)
        self.y_frontier_def = np.bitwise_and(tempisddy1,
                                                self.y_frontier_edge)
        ## we could define only in_corner instead of corner for best performance...
        self.corner_def = np.bitwise_and(temp_y_frontier_def, temp_x_frontier_def)
        self.normal_sign = np.sign(self.nx*self.ny)

        self.isddx1 = conv(self.solid.astype(int), self.ddx1 ** 2) == 2
        self.isddx2 = conv(self.solid.astype(int), self.ddx2 ** 2) == 2
        self.isddy1 = conv(self.solid.astype(int), self.ddy1 ** 2) == 2
        self.isddy2 = conv(self.solid.astype(int), self.ddy2 ** 2) == 2

        self.frontier_def = np.bitwise_or(np.bitwise_not(self.isddx1),
                                             np.bitwise_not(self.isddx2))

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
        a_u_x = (conv(sxx_x,self.ddxx) + conv(sxy_y,self.ddyy)) / self.lm
        a_u_y = (conv(syy_y, self.ddyy) + conv(sxy_x, self.ddxx)) / self.lm

        a_u_x[np.bitwise_not(self.solid)] = 0
        a_u_y[np.bitwise_not(self.solid)] = 0

        a_u_x[self.is_uimp] = 0
        a_u_y[self.is_uimp] = 0

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

    def calc_stress(self,uxt,uyt):
        #Calculate the stress in the center of the mesh edges

        #First, calculate def at mesh cell centers
        duxdx2 = conv(uxt, self.ddx2)*self.isddx2
        duxdy2 = conv(uxt, self.ddy2)*self.isddy2
        duydx2 = conv(uyt, self.ddx2)*self.isddx2
        duydy2 = conv(uyt,self.ddy2) * self.isddy2
        exx = (conv(uxt, self.ddx1)*self.isddx1 +
               duxdx2 ) / (2 *self.lm)
        eyy = (conv(uyt, self.ddy1)*self.isddy1
               + duydy2 ) / (2 * self.lm)
        exy = (conv(uxt, self.ddy1)*self.isddy1 + duxdy2 ) / (4 * self.lm)
        eyx = (conv(uyt, self.ddx1)*self.isddx1 + duydx2 ) / (4 * self.lm)

        #keep those vals in memory for frontier mod later. could be optimised for frontier only...
        exxold = exx
        eyyold = eyy
        exyold = exy
        eyxold = eyx

        # multiply by two on frontiers to compensate where isddx/isddy = 0
        exx[self.y_frontier_def] *= 2
        eyx[self.y_frontier_def] *= 2
        eyy[self.x_frontier_def] *= 2
        exy[self.x_frontier_def] *= 2

        #Adjust exy/eyx on frontiers to have eps_xy=0
        exy[self.y_frontier_def] = -eyx[self.y_frontier_def]
        eyx[self.x_frontier_def] = -exy[self.x_frontier_def]

        # Adjust defs on straight frontier
        coef = - self.elas_lambda / (self.elas_lambda + 2 * self.elas_mu)
        temp=exx[self.y_frontier_def]
        exx[self.x_frontier_def] += coef * eyy[self.x_frontier_def]
        eyy[self.y_frontier_def] += coef * temp#todo temp devrait etre inutile ici

        #adjust defs on in_corners
        #exx=eyy :
        coef = self.elas_mu / (self.elas_lambda + 2 * self.elas_mu)
        coef2 = (2 * self.elas_lambda + 3 * self.elas_mu) / self.elas_mu
        tempxx = coef * ( exx[self.corner_def] + eyy[self.corner_def]
                        - self.normal_sign[self.corner_def] * (exy[self.corner_def]
                                                               + eyx[self.corner_def]))
        tempyx = coef * (self.normal_sign[self.corner_def]*exx[self.corner_def]
                         - exy[self.corner_def]
                         + coef2 * ( eyx[self.corner_def]
                                     - self.normal_sign[self.corner_def] * eyy[self.corner_def]))

        exy[self.corner_def] = coef * (self.normal_sign[self.corner_def] * eyy[self.corner_def]
                                       - eyx[self.corner_def]
                         + coef2 * (exy[self.corner_def]
                                    - self.normal_sign[self.corner_def] * exx[self.corner_def]))

        eyx[self.corner_def] = tempyx
        exx[self.corner_def] = tempxx
        eyy[self.corner_def] = tempxx

        #Average + mod to have def on edges _x perpendicular to x, and _y perpendicular to y
        exx_x = 0.5*conv(exx,self.meany) /2 +0.5*duxdx2 / self.lm
        eyy_x = conv(eyy,self.meany) /2
        exy_x = conv(exy, self.meany) /2
        eyx_x = 0.5*conv(eyx, self.meany) /2 +0.5*duydx2/ self.lm /2

        #duydx2 /2 necessary for exy/eyx because of epsilonxy definition
        exx_y = conv(exx, self.meanx) /2
        eyy_y = 0.5*conv(eyy, self.meanx)/2+0.5*duydy2/ self.lm
        exy_y = 0.5*conv(exy, self.meanx)/2+0.5*duxdy2/ self.lm /2
        eyx_y = conv(eyx, self.meanx)/2

        # Adjust def on frontier to compensate duydy2=0.... todo check
        exx_x[self.x_frontier_edge] += (2*exx_x[self.x_frontier_edge]
                                        - conv(exxold,self.meany)[self.x_frontier_edge]/4)
        eyx_x[self.x_frontier_edge] += (2*eyx_x[self.x_frontier_edge]
                                        - conv(eyxold,self.meany)[self.x_frontier_edge]/4) /2
        eyy_y[self.y_frontier_edge] += (2*eyy_y[self.y_frontier_edge]
                                        - conv(eyyold,self.meanx)[self.y_frontier_edge]/4)
        exy_y[self.y_frontier_edge] += (2*exy_y[self.y_frontier_edge]
                                        - conv(exyold,self.meanx)[self.y_frontier_edge]/4) /2

        #Calculate complete shear deformation exy
        exy_x += eyx_x
        exy_y += eyx_y

        # Adjust shear def on frontier ; should it be done on edges around in_corners ??
        #exy_x[self.x_frontier_edge] = 0
        #exy_y[self.y_frontier_edge] = 0

        #Calculate stress from def
        sxx_x = (self.elas_lambda + 2 * self.elas_mu) * exx_x + self.elas_lambda * eyy_x
        sxy_x =  (2 * self.elas_mu) * exy_x

        syy_y = (self.elas_lambda + 2 * self.elas_mu) * eyy_y + self.elas_lambda * exx_y
        sxy_y =  (2 * self.elas_mu) * exy_y

        #Frontier adjustments
        # TODO : check, it should not be necessary
        # sxx stress is zero on x frontier, same for syy on y frontier
        # sxx_x[self.x_frontier_edge] = 0
        # syy_y[self.y_frontier_edge] = 0

        #Set to zero outside the solid - this one might be necessary
        sxx_x[self.not_solid_x_edge] = 0
        syy_y[self.not_solid_y_edge] = 0
        sxy_x[self.not_solid_x_edge] = 0
        sxy_y[self.not_solid_y_edge] = 0

        return sxx_x,sxy_x,syy_y,sxy_y

