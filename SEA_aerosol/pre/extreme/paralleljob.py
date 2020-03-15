import numpy as np
import multiprocessing

def my_function(var):
    return np.sum(var)
    

X = np.random.normal(size=(3, 10,8))

print(np.sum(X,axis=0))
pool = multiprocessing.Pool(processes=16)

X = X.reshape(3,80)
F=pool.map(my_function, (X[:,i] for i in range(80)) )
F=np.array(F)
F=F.reshape(10,8)
print(F)


X = X.reshape(3,10,8)
F=[]
for i in range(10):
    F = np.concatenate([F,pool.map(my_function, (X[:,i,j] for j in range(8)) ) ])
F=np.array(F)
F=F.reshape(10,8)
print(F)


