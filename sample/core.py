import numpy as np
from scipy.signal import convolve2d,correlate2d
###Remark : pyqt6 needed only for matplotlib show, maybe not necessary for pygame !

def get_frontier(solid):
    # Calculate all the points at the frontier of the solid
    # Frontier = at least one neighbour point is not solid (including diagonals)
    # Bulk is the solid minus the frontier
    kernel = np.ones([3, 3])
    temp = conv(solid, kernel)
    frontier = np.bitwise_and(solid, temp < 8)
    bulk = np.bitwise_and(solid, np.bitwise_not(frontier))

    return frontier, bulk

def remove_single_points(solid):
    # Remove point with no enough neighbours
    # Point needs at least 2 consecutive neighbors to be OK
    kernel = np.array([[1, 1, 0],[0, 0, 0],[0, 0, 0]])
    temp = conv(solid, kernel)
    temp2 = temp>=2

    kernels=[]
    kernels.append(np.array([[0 ,1, 1],[0, 0, 0],[0, 0, 0]]))
    kernels.append(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]))
    kernels.append(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]]))
    kernels.append(np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]]))
    kernels.append(np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]))
    kernels.append(np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]))
    kernels.append(np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]]))

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

def calc_normal_stress(solid):
    # Calculate the normals to the solid boundaries on stress points
    kernelx = np.array([[-1,-1],[1,1]])
    kernely = np.array([[-1,1],[-1,1]])

    nx = conv(solid.astype(int), kernelx)
    ny = conv(solid.astype(int), kernely)
    #n = np.sqrt(nx ** 2 + ny ** 2)
    #ok = n > 0
    #nx[ok] = nx[ok] / n[ok]
    #ny[ok] = ny[ok] / n[ok]
    return nx, ny

def conv(matrix,kernel):
    #return convolve2d(matrix,kernel,'same')
    return correlate2d(matrix, kernel, 'same')


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
        (self.axx,self.axy,self.ayy,self.ayx,
         self.ddx1,self.ddx2,self.ddy1,self.ddy2,
         self.ddxx,self.ddyy) =\
            self.def_kernel()
        self.frontier, self.bulk = get_frontier(self.solid)
        self.nx, self.ny = calc_normal(self.solid)
        self.nx_s, self.ny_s = calc_normal_stress(self.solid)


        self.is_uimp = np.bitwise_and(np.bitwise_not(np.isnan(self.ux_imp)),
                                      np.bitwise_not(np.isnan(self.uy_imp)))
        self.is_uimp = np.bitwise_and(self.is_uimp, self.solid)
        kerneltemp= np.array([[1,1],[1,1]])
        solidtemp = solid.astype(int)
        temp = conv(solidtemp,kerneltemp)
        self.solid_stress = (temp == 4)
        self.in_corner = (temp==3)
        self.in_corner_p = np.bitwise_and(self.in_corner, self.nx_s*self.ny_s > 0)
        self.in_corner_m = np.bitwise_and(self.in_corner, self.nx_s * self.ny_s < 0)

        kernels = []
        kernels.append([[1, 0], [0, 1]])
        kernels.append([[0, 1], [1, 0]])
        self.out_corner = np.zeros(solid.shape).astype(bool)
        for kernel in kernels:
            temp = conv(solid,kernel)
            self.out_corner = np.bitwise_or(self.out_corner,temp==0)

        tempisddx1 = conv(self.solid.astype(int),self.ddx1) != 0
        tempisddx2 = conv(self.solid.astype(int), self.ddx2) != 0

        self.x_frontier_stress = np.bitwise_and(tempisddx1,
                                               tempisddx2)
        tempisddy1 = conv(self.solid.astype(int), self.ddy1) != 0
        tempisddy2 = conv(self.solid.astype(int), self.ddy2) != 0
        self.y_frontier_stress = np.bitwise_and(tempisddy1,
                                               tempisddy2)
        self.isddx1 = conv(self.solid.astype(int), self.ddx1 ** 2) == 2
        self.isddx2 = conv(self.solid.astype(int), self.ddx2 ** 2) == 2
        self.isddy1 = conv(self.solid.astype(int), self.ddy1 ** 2) == 2
        self.isddy2 = conv(self.solid.astype(int), self.ddy2 ** 2) == 2

    def def_kernel(self):
        if self.kernel_type=='plane strain':
            axx=np.zeros([3,3])
            axy= np.zeros([3,3])
            ayy=np.zeros([3,3])
            ayx=np.zeros([3,3])

            temp=(self.elas_lambda+2*self.elas_mu)/self.lm**2
            axx[2,1] += temp
            axx[0,1] += temp
            axx[1,1] += -2*temp
            temp = temp/2
            axy[2,2] += temp
            axy[2,0] -= temp
            axy[0,2] -= temp
            axy[0,0] += temp
            temp = self.elas_mu / self.lm**2
            axx[1,2] += temp
            axx[1,0] += temp
            axx[1,1] += -2*temp

            ayy = np.transpose(axx)
            ayx = axy

            #Mesh cell center stress matrix
            # epsilon_xx = ddx * ux / (2 lm)
            # epsilon_yy = ddy * uy / (2 lm)
            # epsilon_xy = (ddy * ux + ddx * uy) / (4 lm)
            ddx1 = np.array([[-1, 0], [1, 0]])
            ddx2 = np.array([[0, -1], [0, 1]])
            ddy1 = np.array([[-1,1],[0,0]])
            ddy2 = np.array([[0, 0], [-1, 1]])

            ddxx = np.array([[-1, -1, 0], [1, 1, 0], [0, 0, 0]])
            ddyy = np.transpose(ddxx)
        else:
            raise ValueError("Only \'plane strain\' is available")
        return axx,axy,ayy,ayx,ddx1,ddx2,ddy1,ddy2,ddxx,ddyy

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

        sxx,syy,sxy = self.calc_stress(uxt,uyt)


        a_u_x = conv(sxx,self.ddxx) + conv(sxy,self.ddyy)
        a_u_y = conv(syy, self.ddyy) + conv(sxy, self.ddxx)

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
        #Calculate the stress in the center of the mesh cells
        exx = (conv(uxt, self.ddx1)*self.isddx1 +
               conv(uxt, self.ddx2)*self.isddx2 ) / (2 *self.lm)
        eyy = (conv(uyt, self.ddy1)*self.isddy1
               + conv(uyt,self.ddy2) * self.isddy2 ) / (2 * self.lm)
        exy = (conv(uxt, self.ddy1) + conv(uxt, self.ddy2) +
               conv(uyt, self.ddx1) + conv(uyt, self.ddx2)   ) / (4 * self.lm)

        # We could multiply by 2 exx/eyy on x/y frontiers respectively, depending on how
        # we want to handle frontiers
        exx[self.y_frontier_stress] = 2 * exx[self.y_frontier_stress]
        eyy[self.x_frontier_stress] = 2 * eyy[self.x_frontier_stress]

        # Special formulas on frontiers for exx/eyy to be OK with sxx/syy=0
        coef = - self.elas_lambda / (self.elas_lambda + 2*self.elas_mu)
        exx[self.x_frontier_stress] = coef * eyy[self.x_frontier_stress]
        eyy[self.y_frontier_stress] = coef * exx[self.y_frontier_stress]

        # In inside corners, exx=eyy = mean (exx,eyy,exy* (+/-) mu / (lambda+mu))*2
        # does not work well
        # temp = 2/3*((exx[self.in_corner_p] + eyy[self.in_corner_p])*(1+coef) -
        #         self.elas_mu / (self.elas_lambda + self.elas_mu) * exy[self.in_corner_p])
        # exx[self.in_corner_p] = temp
        # eyy[self.in_corner_p] = temp
        # exy[self.in_corner_p] = - (self.elas_lambda + self.elas_mu) / (self.elas_mu) * exx[self.in_corner_p]
        #
        # temp = 2/3*((exx[self.in_corner_m] + eyy[self.in_corner_m]) * (1 + coef) +
        #         self.elas_mu / (self.elas_lambda + self.elas_mu) * exy[self.in_corner_m])
        # exx[self.in_corner_m] = temp
        # eyy[self.in_corner_m] = temp
        # exy[self.in_corner_m] = (self.elas_lambda + self.elas_mu) / (self.elas_mu) * exx[self.in_corner_m]

        # Another strategy for test :
        #temp = exx[self.in_corner]
        #exx[self.in_corner] = (exx[self.in_corner] + coef*eyy[self.in_corner])
        #eyy[self.in_corner] = (eyy[self.in_corner] + coef*temp)
        #exy[self.in_corner] =  exy[self.in_corner]

        # Another one :
        temp = exx[self.in_corner]
        exx[self.in_corner] = (2*exx[self.in_corner] + coef*eyy[self.in_corner])/3
        eyy[self.in_corner] = (2*eyy[self.in_corner] + coef*temp)/3
        exy[self.in_corner] =  exy[self.in_corner]*2/3
        #test exy from exx/eyy

        lambda_trace = self.elas_lambda * (exx+eyy)
        sxx = lambda_trace + (2 * self.elas_mu) * exx
        syy = lambda_trace + (2 * self.elas_mu) * eyy
        sxy = (2 * self.elas_mu) * exy

        #Frontier adjustments
        # sxx stress is zero on x frontier, same for syy on y frontier

        sxx[self.x_frontier_stress] = 0
        syy[self.y_frontier_stress] = 0
        sxy[self.x_frontier_stress] = 0
        sxy[self.y_frontier_stress] = 0

        sxx[self.out_corner] = 0
        syy[self.out_corner] = 0
        sxy[self.out_corner] = 0

        return sxx,syy,sxy

