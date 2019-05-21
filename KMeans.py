
"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA


def NIUs():
    return 1458626, 1111112, 1111113
    
def distance(X,C):
    """@brief   Calculates the distance between each pixel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
    return np.array([[np.linalg.norm(c-p) for c in C] for p in X])

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
        self.X = X.reshape(np.prod(X.shape[:-1]), 3)

            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if self.K>0:
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        self.num_iter = 0                                      # INT current iteration
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        if self.options['km_init'].lower() == 'first':
                self.centroids = np.array(list({tuple(row) for row in self.X})[:self.K])
        elif self.options['km_init'].lower() == 'random':
	        self.centroids = np.random.rand(self.K,self.X.shape[1])
        if self.options['km_init'].lower() == 'spaced':
                self.centroids=np.zeros((self.K, self.X.shape[1]))
                for k in range(self.K): self.centroids[k, :] = k*255/(self.K-1)
        
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
        #self.clusters = np.random.randint(self.K,size=self.clusters.shape)
        self.clusters = np.array([[np.linalg.norm(c-p) for c in self.centroids] for p in self.X]).argmin(axis=1)

        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        self.centroids = np.array([np.mean(self.X[np.where(self.clusters==c)], axis=0) for c in range(len(self.centroids))])
                

    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        return not np.any(np.linalg.norm(self.centroids-self.old_centroids, axis=1)>self.options['tolerance'])
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K==0:
            self.bestK()
            return        
        
        self._iterate(True)
        if self.options['max_iter'] > self.num_iter:
            while not self._converges():
                self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
        K = 1
        self._init_rest(K)
        self.run()
        fit = self.fitting()
        ant = fit
        ant2 = ant
        while (ant-fit)*2 < (ant2 - ant):
            self._init_rest(K)
            self.run()
            ant2 = ant
            ant = fit
            fit = self.fitting()
            K += 1
        return K
        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        if self.options['fitting'].lower() == 'fisher':
            intra = np.array([[np.linalg.norm(c-p)/(np.sum(self.clusters==c)) for c in self.centroids] for p in self.X]).sum()/self.K
            center = np.mean(self.X,axis=0)
            inter = np.array([np.linalg.norm(c-center) for c in self.centroids]).sum()/self.K
            return intra/inter
        elif self.options['fitting'].lower() == 'silhouette':
            return np.random.rand(1)


    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'	
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
