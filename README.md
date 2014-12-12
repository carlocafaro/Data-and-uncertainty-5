Data-and-uncertainty-5
======================
import pylab as pl
import numpy as np
from scipy import integrate

def Vprime_bistable(x):
    return x**3-x

def V_bistable(x):
    return x**4/4-x**2/2

def exact_solution(x):
    integrand= lambda x: np.exp((-2/sigma**2)*V_bistable(x))
    return 1/(integrate.quad(integrand, -1*np.inf, np.inf))[0]*np.exp(-2/(sigma**2)*V_bistable(x))  

def analytic_expectedvalue(f,x):    
    integranda= lambda x: exact_solution(x)*f(x)
    return integrate.quad(integranda, -1*np.inf, np.inf)[0]    


   
def identity(x): 
    return x

def square(x):
    return x**2


def brownian_dynamics(x,Vp,sigma,dt,T,tdump):
    """
    Solve the Brownian dynamics equation

    dx = -V(x)dt + sigma*dW

    using the Euler-Maruyama method

    x^{n+1} = x^n - V(x^n)dt + sigma*dW

    inputs:
    x - initial condition for x
    Vp - a function that returns V'(x) given x
    dt - the time stepsize
    T - the time limit for the simulation
    tdump - the time interval between storing values of x

    outputs:
    xvals - a numpy array containing the values of x at the chosen time points
     tvals - a numpy array containing the values of t at the same points
    """

    xvals = [x]
    t = 0.
    tvals = [0]
    dumpt = 0.
    while(t<T-0.5*dt):
        dW = dt**0.5*np.random.randn()
        x += -Vp(x)*dt + sigma*dW

        t += dt
        dumpt += dt
        if(dumpt>tdump-0.5*dt):
            dumpt -= tdump
            xvals.append(x)
            tvals.append(t)
    return np.array(xvals), np.array(tvals)

    
if __name__ == '__main__':
    dt = 0.01
    sigma =0.5
    T = 10000.0
    N= 100  
    'N is the number of samples'
    a=np.zeros((N, T/dt+1))  
    'This creates an 2D array which contains N samples at each time'
    
    Trange=np.array([0, 100, 200, 500 , 1000, 5000])  
    'This is to visualize the solution at time 0, 100 and so on'

  
    for i in range(N):
          a[i,:] = brownian_dynamics(1.0,Vprime_bistable,sigma,dt,T,dt)[0]


    a = a[:,Trange]
    #Computationa Lab Dynamics 5, Exercise 2   
    x=np.linspace(-2,2,100)
    for j in range(6):
        pl.plot(x, exact_solution(x)) 
        pl.hist(a[:,j],50 ,normed=1)
        pl.savefig('bistable.pdf')   
        pl.show()
     
    ' How quickly does the solution converge to the steady state solution? (Make a qualitative answer.)'
     
    'As we can see from the graph the solution converges to the steady state solution quickly. For example at time t=0 obviously the    solution is concentrated in x=1.0 but at time t=10.0 the solution has already almost the same shape of the steady state'

   
   #Computational Lab Dynamics 5, Exercise 3
    Ns=np.linspace(1000, 1000000, 100).astype(int)
    b=np.cumsum(square(brownian_dynamics(1.0, Vprime_bistable, sigma, dt, 10000.0, dt)[0])) 
    c=b/np.arange(1,b.size+1)
    d=c[Ns] 
    err=np.absolute(d-analytic_expectedvalue(square,x)) 
    pl.plot(Ns,err)
    pl.show()
   
    'As we can see from the loglog graph the convergence of the error is exponential'

