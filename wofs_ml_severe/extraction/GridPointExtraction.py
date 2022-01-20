import numpy as np

class PatchExtraction: 
    
    def __init__(self, grid_size=20):
        '''
        '''
        self.grid_size = grid_size
        self.delta = int( grid_size/2)

    def subsample(self, observations):
        '''
        Subsample the 
        '''
        array = np.ones(observations.shape)*-1000.
        array[self.delta:array.shape[0]-self.delta, self.delta:array.shape[1]-self.delta] = observations[self.delta:array.shape[0]-self.delta, self.delta:array.shape[1]-self.delta]   

        # Randomly sample the positive class grid points
        pos_j,pos_i = np.where((array>0))
        random_idx = list(np.random.choice(np.arange(len(pos_j)), size=int(0.5*len(pos_j)), replace=False))
        pos_j_random = list(np.array(pos_j)[random_idx])
        pos_i_random = list(np.array(pos_i)[random_idx])

         # Randomly sample the negative class grid points
        neg_j,neg_i = np.where((array==0))
        random_idx = list(np.random.choice(np.arange(len(neg_j)), size=int(0.01*len(neg_j)), replace=False))
        neg_j_random = list(np.array(neg_j)[random_idx]) 
        neg_i_random = list(np.array(neg_i)[random_idx])        

        labels = [1]*len(pos_j_random) + [0]*len(neg_j_random) 
        
        return zip( pos_j_random+neg_j_random, pos_i_random+neg_i_random), labels

    def extract_patch(self, data, centers):
        '''
        Extract patches
        data (y,x,v)
        '''
        storm_patches = [ ]
        for obj_y, obj_x in centers:
            storm_patches.append( data[:, obj_y-self.delta:obj_y+self.delta, obj_x-self.delta:obj_x+self.delta] )

        return storm_patches

    


