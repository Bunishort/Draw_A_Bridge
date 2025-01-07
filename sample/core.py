import numpy as np
from scipy.signal import convolve2d

def conv2(matrix, kernel):
    '''
    :param matrix: 2D numpy matrix
    :param kernel: 2D numpy matrix, smaller than matrix in each direction
    :return: conv : convoluted matrix
    '''
    import numpy as np
    conv = np.convolve(matrix,kernel)
    return conv

class ElasticProblem:
    """
    :param solid: Bool 2d matrix containing the position of the solid on the grid (1 if solid, 0 if not)
    :param elas_lambda
    :param elas_mu : Lamé elastic coefficients
    :param lm : pixel length
    :param ux_imp : 2d matrix of imposed displacements in the x direction. np.Nan where no displacement is imposed
    :param uy_imp : 2d matrix of imposed displacements in the y direction np.Nan where no displacement is imposed
    :param px_bound : 2d matrix of imposed stress on solid boundary in the x direction (sig.n = (px_bound,py_bound))
    :param py_bound : 2d matrix of imposed stress on solid boundary in the y direction
    :param fx_imp : 2d matrix of imposed volumic forces in the x direction
    :param fy_imp : 2d matrix of imposed volumic forces in the y directions
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
        for var in ['px_bound,py_bound,fx_imp,fy_imp']:
            setattr(self,var,kwargs.get(var,np.zeros(solid.shape)))
        for var in  ['ux','uy']:
            setattr(self,var, np.zeros(solid.shape))
        self.kernel_type = 'plane strain'
        (self.axx,self.axy,self.ayy,self.ayx,
         self.exxx,self.eyyy,self.exyx,self.exyy) =\
            self.def_kernel()
        self.frontier, self.bulk = self.get_frontier()
        self.nx, self.ny = self.calc_normal()



    def conv(self,matrix,kernel):
        return convolve2d(matrix,kernel,'same')

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
            # epsilon_xx = exxx * ux / (2 lm)
            # epsilon_yy = eyyy * uy / (2 lm)
            # epsilon_xy = (exyx * ux + exyy * uy) / (4 lm)
            exxx = np.array([[-1,-1],[1,1]])
            eyyy = np.array([[-1,1],[-1,1]])
            exyx = eyyy.copy()
            exyy = exxx.copy()
        else:
            raise ValueError("Only \'plane strain\' is available")
        return axx,axy,ayy,ayx,exxx,eyyy,exyx,exyy

    def calc_stress(self,uxt,uyt):
        #Calculate the stress in the center of the mesh cells

        exx = self.conv(uxt, self.exxx) / (2 *self.lm)
        eyy = self.conv(uyt, self.eyyy) / (2 * self.lm)
        exy = (self.conv(uxt, self.exyx) +
               self.conv(uyt, self.exyy)) / (4 * self.lm)
        lambda_trace = self.elas_lambda * (exx+eyy)
        sxx = lambda_trace + (2 * self.elas_mu) * exx
        syy = lambda_trace + (2 * self.elas_mu) * eyy
        sxy = (2 * self.elas_mu) * exy

        return sxx,syy,sxy

    def get_frontier(self):
        #Calculate all the points at the frontier of the solid
        # Frontier = at least one neighbour point is not solid (including diagonals)
        # Bulk is the solid minus the frontier
        kernel = np.ones([3,3])
        temp = self.conv(self.solid,kernel)
        frontier = np.bitwise_and(self.solid,temp < 9)
        bulk = np.bitwise_and(self.solid,np.bitwise_not(frontier))

        return frontier,bulk

    def calc_normal(self):
        #Calculate the normals to the solid boundaries
        kernelx = np.zeros([3,3])
        kernelx[2,1] = 1
        kernelx[0,1] = -1
        kernely = np.zeros([3,3])
        kernely[1,2]=1
        kernely[1,0]=-1

        nx = self.conv(self.solid, kernelx)
        ny = self.conv(self.solid, kernely)
        n = np.sqrt(nx**2 + ny **2)
        ok = n>0
        nx[ok] = nx[ok] / n[ok]
        ny[ok] = ny[ok] / n[ok]
        return nx,ny

    def CG_loop(self):
        """
        :return: ux, uy : Displacements solutions
        :return: n_iter : number of iterations
        :return: res : residual

        Solve a_u = b
        """
        n_iter = 0
        res = np.zeros(self.solid.shape)
        ux = np.zeros(self.solid.shape)
        uy = np.zeros(self.solid.shape)



        return ux,uy,n_iter,res

    def calc_a_u(self,uxt,uyt):
        #In the bulk, a_u = div(sigma)
        #On the frontier, a_u = sigma.n
        #Elsewhere, it is 0
        #uxt,uyt are 2d matrices of displacements

        a_u_x = self.conv(uxt,self.axx) + self.conv(uyt,self.axy)
        a_u_y = self.conv(uyt,self.ayy) + self.conv(uxt,self.axy)

        sxx,syy,sxy = self.calc_stress(uxt,uyt)

        a_u_x[self.frontier] = (sxx[self.frontier]*self.nx[self.frontier]
                                     + sxy[self.frontier]*self.ny[self.frontier])
        a_u_y[self.frontier] = (sxy[self.frontier] * self.nx[self.frontier]
                                     + syy[self.frontier] * self.ny[self.frontier])

        a_u_x[np.bitwise_not(self.solid)] = 0
        a_u_y[np.bitwise_not(self.solid)] = 0

        return a_u_x,a_u_y

    def calc_b(self):
        #In the bulk, b= fx_imp, fy_imp (volumic forces, no inertia taken into account at this stage)
        #On the frontier, b = px_bound,py_bound
        #Elsewhere, b=0

        bx = self.fx_imp
        by = self.fy_imp
        bx[self.frontier] = self.px_bound[self.frontier]
        by[self.frontier] = self.py_bound[self.frontier]

        return bx,by