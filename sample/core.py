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
    axx: kernel _x_x for div(sigma)
    axy: kernel _x_y for div(sigma)
    ayy: kernel _y_y for div(sigma)
    ux: Initial guess for x displacements (2d numpy matrix, same size as solid)
    uy: Initial guess for y displacements (2d numpy matrix, same size as solid)
    ux_imp : 2d matrix of imposed displacements in the x direction
    uy_imp : 2d matrix of imposed displacements in the y direction
    sigx : 2d matrix of imposed stress on solid boundary in the x direction (sig.n = (sigx,sigy))
    sigy : 2d matrix of imposed stress on solid boundary in the y direction
    fx_imp : 2d matrix of imposed volumic forces in the x direction
    fy_imp : 2d matrix of imposed volumic forces in the y directions
    """
    def __init__(self,solid,elas_lambda,elas_mu,lm):
        self.solid=solid
        self.elas_lambda = elas_lambda
        self.elas_mu =elas_mu
        self.lm = lm
        for var in  ['ux','uy','ux_imp','uy_imp', 'sigx','sigy','fx_imp','fy_imp']:
            setattr(self,var, np.zeros(solid.shape))
        self.kernel_type = 'plane strain'
        (self.axx,self.axy,self.ayy,self.ayx,
         self.exxx,self.eyyy,self.exyx,self.exyy) =\
            self.def_kernel()

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

    def calc_stress(self):
        #Calculate the stress in the center of the mesh cells

        exx = self.conv(self.ux, self.exxx) / 2 / self.lm
        eyy = self.conv(self.uy, self.eyyy) / 2 / self.lm
        exy = (self.conv(self.ux, self.exyx) +
               self.conv(self.uy, self.exyy)) / 4 / self.lm
        trace = exx+eyy
        sxx = self.elas_lambda * trace +2*self.elas_mu * exx
        syy = self.elas_lambda * trace + 2 * self.elas_mu * eyy
        sxy = 2 * self.elas_mu * exy

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
        ny = self.conv(self.solid,kernely)
        n = np.sqrt(nx**2 + ny **2)
        nx = nx / n
        ny = ny /n
        return nx,ny

    def CG_loop(self):
        """
        :return: Ux, Uy : Displacements solutions
        :return: n_iter : number of iterations
        :return: res : residual
        """
        n_iter = 0
        res = np.zeros(self.solid.shape)
        ux = np.zeros(self.solid.shape)
        uy = np.zeros(self.solid.shape)

        return ux,uy,n_iter,res