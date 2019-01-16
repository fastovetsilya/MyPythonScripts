import numpy as np
try: 
    del A, A_sq, Id, n_iter
except:
    pass

A = np.random.randint(-10, 10, (2,2))
#A = np.array([[0,1], 
#             [-1, 0]])
A_sq = A @ A
Id = np.array([[-1, 0],
               [0, -1]])
n_iter = 1

while not np.all(np.equal(A_sq, Id)):
#    if np.all(np.equal(A_sq, np.zeros((2,2), dtype=int))) == True:
#        continue
    A = np.random.randint(-2, 2, (2,2))
    A_sq = A @ A
    n_iter += 1

print(A)
print(n_iter)
