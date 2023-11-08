#prep
import numpy as np
#set values of N and K 
N = 3
K = 2
X = 2**(K+1) #total fitness contributions per gene (= columns in fitness matrix)

#create fitness matrix: np.array of shape (N,X), filled with random decimals 
fm = np.random.rand(N, X)
print(fm)

#calculating coefficients
def calc_a(K, fm):
    a_coef = []
    #explain for loop
    for r in fm:
        a = [0.0] * X  # creates list with zeros as floats for each row & X cols
        
        # Calculate ai0 for i = 0
        a[0] = r[0] #because ai0=Fi0
        
        for j in range(1, X): 
            sum = 0.0 
            for l in range(0, j): #only already calculated coeff
                if l == (l & j): #if l equal to bitwise AND of l and j (001&101->TRUE, 001&100->FALSE)
                    sum += a[l] 
            a[j] = r[j] - sum 
        a_coef.append(a) # append new a's into a_values array
        
    return a_coef

a_coef = calc_a(K, fm)

a_shape = np.reshape(a_coef, (N,X))
print("Fitness matrix")
print(a_shape) #all coefficients in same shape as fm1

#To-do: maybe create (example) pd dataframe with all binary id's and coefficients
im0 = np.arange(0, X, 1)
im1 = im0[np.newaxis, :]
im = np.repeat(im1, N, axis=0)
#Binary representation of im (just for visualisation)
imbin = np.vectorize(np.binary_repr)(im, 4) #increase  to 8/16/32 with larger N 
print("identity matrix (maybe useful for vis)")
print(imbin)
#check if shape is same
print(a_shape.shape)
print(imbin.shape)
#make pd dataframe

#manual check
#Set model N to 3 and K to 2 for check
Fi0 = (fm[:, [0,]])
ai0 = Fi0
#ai(1)
Fi1 = (fm[:, [1,]])
ai1 = Fi1 - ai0
#ai(2)
Fi2 = (fm[:, [2,]])
ai2 = Fi2 - ai0
#ai(3)
Fi3 = (fm[:, [3,]])
ai3 = Fi3 - ai0 - ai1 - ai2
#ai(4)
Fi4 = (fm[:, [4,]])
ai4 = Fi4 - ai0
#ai(5)
Fi5 = (fm[:, [5,]])
ai5 = Fi5 - ai0 - ai4 - ai1
#ai(6)
Fi6 = (fm[:, [6,]])
ai6 = Fi6 - ai0 - ai2 - ai4
#ai(7)
Fi7 = (fm[:, [7,]])
ai7 = Fi7 - ai0 - ai1 - ai2 - ai3 - ai4 - ai5 - ai6

#check if coefficients match
print("manual a")
print(ai3) #any a
print("coefficient array from model")
print(a_shape)
#if same model works :)

#Introducing neutrality

#NKp: reduce fraction of fitness contributions in fm to 0
print("NK coefficients without neutrality")
print(fm)

# Number of elements to replace
p = ((N-1)/N)   
num_p = int(((N-1)/N) * (N*X)) #(with p = (N-1)/N (Geard 10.1109/CEC.2002.1006234))

# Random (x, y) coordinates
indices_x = np.random.randint(0, fm.shape[0], num_p)
indices_y = np.random.randint(0, fm.shape[1], num_p)
fmp = fm[indices_x, indices_y] = 0
print("NKp coefficients")
print(fmp)

#NKq: fitness contribution < 0.5 == 0, > 0.5 == 1
fmq1 = np.where(fm > 0.5, 1.0, fm)
fmq = np.where(fmq1 < 0.5, 0.0, fmq1)
print("NKq coefficients")
print(fmq) # = new, modified fm