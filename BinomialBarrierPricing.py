# %%
# 5220 Group Project: Barrier Option Pricing
# Kyle Nunn

# Exotic Option: Barrier up and out european option pricing
%cd /Users/kylenunn/Desktop/Machine-Learning
!pwd

# %%
# Looking at a barrier up and out call option with the following variables:
# Strike Price = $100
# Barrier = $150
# Rebate = $50
# Exercise date = 4 years

# %%
# We are going to do a simple slow and fast binomial pricing model in python.
# We will treat binomial tree as a network with nodes (i,j)
# i represents the time steps
# j represents the number of ordered price outcome (bottom of tree to top of tree)
# binomial_tree_slow
# binomial_tree_fast

# %%
import numpy as np

# %%
# Binomial tree representation
# Stock tree can be represented using nodes (i,j) with initial stock price S0
## *INSERT REPRESENTATION SOMEHWERE* ##

# %%
S0 = 100 # initial stock price
K = 100 # strike price
T = 1 # time to maturity in years
H = 125 #up and out barrier price
r = .06 # risk free rate
N = 3 # number of time steps
u = 1.1 #up factor
d = 1/u #down factor
opttype = 'C' # option price call or put

# %%
# Slow binomial option price model
# Its going to be slow because we are going to go through each node as we did in excel as a class in excel

def barrier_tree_slow(K,T,S0,H,r,N,u,d,opttype='C'):
    #precompute value
    dt = T/N #this is the change each time step
    q = (np.exp(r*dt) - u)/(u-d) # risk neutral probability that we used in bopm
    disc = np.exp(-r*dt)# discounted rate
    
    #initialize asset prices at maturity
    S = np.zeros(N + 1) #setting an empty numpy array with dimensions 0+1
    for j in range(0,N+1):
        S[j] = S0 * u**j * d**(N-j) # Represented by barrier option pricing formula above
   
    # One we initialize our stock prices we can now consider our option payoff
    # option payoff
    # considering a call
    C = np.zeros(N+1)
    for j in range(0,N+1):
        if opttype == 'C':
            C[j] = max(0,S[j]- K)
        else:
            C[j] = max(0,K - S[j]) 
    # We know this is a barrier option, so it might not be the case
    # Therefore, we need to check to see if our barrier has been exceeded
    # So we'll check terminal condition payoff
    for j in range(0,N+1):
        S = S0 * u**j * d**(N-j) # Copying above formula
        if S >= H:
            C[j] = 0 # If the barrier has been exceeded, then the value will be 0 and we lose our option rights        

# Now we will work backward through the option tree to figure out price today
    for i in np.arange(N-1,-1,-1): # starting on the time before the final payoff
        for j in range(0,i+1): #its i+1 because at the second last stage we are going to have - being n-1 and that means that it actually has nj nodes (j in terms of time)
            S = S0 * u**j * d**(N-j)
            if S >= H:
                C[j] = 0
            else:
                C[j] = disc * (q*C[j+1]+(1-q)*C[j]) # C[j] is going to the be risk neutral discounted expectation
    return C[0]

barrier_tree_slow(K,T,S0,H,r,N,u,d,opttype='P')

# %%
# Now that we've done the barrier implementation, we are going to look and see how we can speed things up using the fact method
# We are going to do this by using numpy arrays
# We're going to copy the above code and get rid of the for loops
# We can leave the precomputed values the same
# The initialized assets can now be calculated in one step:

# FIRST, we are going to get rid of the for loops

# We can leave the precomputed values the same:
def barrier_tree_fast(K,T,S0,H,r,N,u,d,opttype='C'):
    dt = T/N 
    q = (np.exp(r*dt) - u)/(u-d) 
    disc = np.exp(-r*dt)
    
    #initialize asset prices at maturity
    # Were going to use numpy arrays within the computation itself
    S = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1)) # Taking "N" steps down by one each time
   
    # One we initialize our stock prices we can now consider our option payoff
    # option payoff
    # considering a call
    # instead of using max we will use maximum. This will compare the values of the array to whatever function we say and we're comparing the zeroes
    if opttype == 'C':
        C = np.maximum(S - K,0)
    else:
        C = np.maximum(K - S,0) 
    
    # We know this is a barrier option, so it might not be the case
    # Therefore, we need to check to see if our barrier has been exceeded
    # So we'll check terminal condition payoff
    # We just need to use the functionality of having a numpy array and we can condition on it by indexing
    # Taking C array and becuase we know the S has been indexed at the same place of all the call or derivative values then we can condition on S with respect to H 
    C[S >= H] = 0
       
# Now we will work backward through the option tree to figure out price today
# This part will change completely from the above short method
    for i in np.arange(N-1,-1,-1): # starting on the time before the final payoff
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1)) # Instead of N we are going to use i becuase the actual column is changing each time.
        C[:i+1] = disc * ( q * C[1:i+2] + (1-q)* C[0:i+1]) # Calculating this in terms of i+1 because we've started from N-1
        C = C[:-1] #There is one quirk which the above arrays need to be the same size. They aren't now becuase S is decreasing in size while C is staying the same.We're chopping off the top value. We don't need it becuase we're keeping our fixed array and the important values are just going down to the bottom.
        C[S >= H] =0  #Now conditioning C onto S just like we did with terminal condition
    return C[0]

barrier_tree_fast(K,T,S0,H,r,N,u,d,opttype='P')