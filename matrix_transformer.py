class Mariamz_Transformer ():

    def __init__ ( self , X):
        self.X = X 
        
    def col_mean(self):
                            #mean of each column (feature) separately
        from functools import reduce
        import numpy as np
        mean=[]
        for i in range(0,(self.X.shape[1]-1)):
            mean.append((reduce(lambda x,y:x+y,self.X[i]))/np.size(self.X[i]))
        return mean
    
    def col_std(self):  
                           #standard deviation of each column (feature) separately
        from functools import reduce
        import numpy as np
        STD=[]
        for i in range(0,(self.X.shape[1]-1)):
            STD.append((reduce(lambda x,y:x+y,(list(map((lambda x:(x-self.col_mean()[i])**2),self.X[i])))))/np.size(self.X[i]))
        return STD
    
     
    def standardization(self):
                                #transform each column(feature) by 
                                #subtracting the column mean from each value in the column and 
                                #dividing the result by the column standard deviation 
        import numpy as np
        Z=[]
        for i in range(0,(self.X.shape[1]-1)):
             Z.append((self.X[i]-self.col_mean()[i])/self.col_std()[i])
        return np.array(Z)
    
    
    
    def normalization(self):
                                #transform each column(feature) by 
                                #for each column, subtracting the minimum value of the column  
                                #from each value in the column and 
                                #dividing the result by the result of 
                                #subtracting the minimum from the maximum value of the column
        import numpy as np
        N=[]
        for i in range(0,(self.X.shape[1]-1)):
            N.append((X[i]-np.min(X[i]))/(np.max(X[i])-np.min(X[i])))
        return np.array(N)
            
