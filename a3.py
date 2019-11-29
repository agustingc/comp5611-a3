#!/usr/bin/env python

from numpy import array, arange, zeros,float_,sum,prod,shape,diagonal,dot,argmax,sin,sign

'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''

def python2_vs_python3():
    '''
    A few of you lost all their marks in A2 because their assignment contained
    Python 2 code that does not work in Python 3, in particular print statements
    without parentheses. For instance, 'print hello' is valid in Python 2 but not
    in Python 3.
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer **with Python 3** and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding Python 2 syntax to this file...
    Test: 'tests/test_python3.py'
    '''
    return ("I won't use Python 2 syntax my code",
            "I will always use parentheses around print statements ",
            "I will check the outcome of the tests using pytest or Travis"
            )

'''
    Part 2: Integration (Chapter 6)
'''

def problem_6_1_18(x):
    '''
    We will solve problem 6.1.18 in the textbook.
    Task: The function must return the integral of sin(t)/t 
          between 0 and x:
              problem_6_1_18(x) = int_0^x{sin(t)/t dt}
    Example: problem_6_1_18(1.0) = 0.94608
    Test: Function 'test_problem_6_1_18' in 'tests/test_problem_6_1_18.py'
    Hints: * use the composite trapezoid rule
           * the integrated function has a singularity in 0. An easy way
             to deal with this is to integrate between a small positive value and x.
    '''

    ## YOUR CODE HERE
    def f_6_1_18(t):
        return sin(t)/t
    #return result
    return  trapezoid_rec(f_6_1_18,10E-12,x) #calls recursuve trapezoidal method
    raise Exception("Not implemented")


def example_6_12():
    '''
    We will implement example 6.12 in the textbook:
        "
            Evaluate the value of int_1.5^3 f(x)dx ('the integral of f(x)
            between 1.5 and 3'), where f(x) is represented by the 
            unevenly spaced data points defined in x_data and y_data.
        "
    Task: This function must return the value of int_1.5^3 f(x)dx where 
          f(x) is represented by the evenly spaced data points in x_data and 
          y_data below.
    Test: function 'test_example_6_12' in 'tests/test_example_6_12.py'.
    Hints: 1. interpolate the given points by a polynomial of degree 5. 
           2. use 3-node Gauss-Legendre integration (with change of variable)
              to integrate the polynomial.
    '''
    
    x_data = array([1.2, 1.7, 2.0, 2.4, 2.9, 3.3])
    y_data = array([-0.36236, 0.12884, 0.41615, 0.73739, 0.97096, 0.98748])
    ## YOUR CODE HERE
    I = gauss_leg_n2(1.5,3,x_data,y_data) #returns Gauss quadrature data points abcissas
    return I
    raise Exception("Not implemented")


'''
    Part 3: Initial-Value Problems
'''


def problem_7_1_8(x):
    '''
    We will solve problem 7.1.8 in the textbook. A skydiver of mass m in a 
    vertical free fall experiences an aerodynamic drag force F=cy'² ('c times
    y prime square') where y is measured downward from the start of the fall, 
    and y is a function of time (y' denotes the derivative of y w.r.t time).
    The differential equation describing the fall is:
         y''=g-(c/m)y'²
    And y(0)=y'(0)=0 as this is a free fall.
    Task: The function must return the time of a fall of x meters, where
          x is the parameter of the function. The values of g, c and m are
          given below.
    Test: function 'test_problem_7_1_8' in 'tests/test_problem_7_1_8.py'
    Hint: use Runge-Kutta 4.
    '''
    g = 9.80665  # m/s**2
    c = 0.2028  # kg/m
    m = 80  # kg

    ## YOUR CODE HERE
    def F_7_18(x, y):
        return array([
            y[1],
            g - (c/m) * y[1]**2
        ], dtype=float_)

    x0=0 #initial value of time, i.e. t=0
    y0 = array([0,0], dtype=float_)   #initial values
    h = 0.01 #suggested by professor
    X, Y = runge_kutta_4_modified(F_7_18,x0,y0,x,h) #here, x is the limit of the y variable (see runge kutta implementation code)
    return X[-1]  #returns last element of X (not Y!)

    raise Exception("Not implemented")



def problem_7_1_11(x):
    '''
    We will solve problem 7.1.11 in the textbook.
    Task: this function must return the value of y(x) where y is solution of the
          following initial-value problem:
            y' = sin(xy), y(0) = 2
    Test: function 'test_problem_7_1_11' in 'test/test_problem_7_1_11.py'
    Hint: Use Runge-Kutta 4.
    '''

   ## YOUR CODE HERE
    def F(x,y):
        return array([ sin(x*y) ], dtype=float_)

    x0=0 #we will plot solutions in the range 0 to 10
    y0=array([2.0],dtype=float_)
    h = 0.01
    _ , Y = runge_kutta_4(F,x0,y0,x,h)
    return Y[-1,0]
    raise Exception("Not implemented")

'''
    Part 4: Two-Point Boundary Value Problems
'''

def problem_8_2_18(a, r0):
    '''
    We will solve problem 8.2.18 in the textbook. A thick cylinder of 
    radius 'a' conveys a fluid with a temperature of 0 degrees Celsius in 
    an inner cylinder of radius 'a/2'. At the same time, the outer cylinder is 
    immersed in a bath that is kept at 200 Celsius. The goal is to determine the 
    temperature profile through the thickness of the cylinder, knowing that
    it is governed by the following differential equation:
        d²T/dr²  = -1/r*dT/dr
        with the following boundary conditions:
            T(r=a/2) = 0
            T(r=a) = 200
    Task: The function must return the value of the temperature T at r=r0
          for a cylinder of radius a (a/2<=r0<=a).
    Test:  Function 'test_problem_8_2_18' in 'tests/test_problem_8_2_18'
    Hints: Use the shooting method. In the shooting method, use h=0.01 
           in Runge-Kutta 4.
    '''

    ## YOUR CODE HERE
    def F_8_2_18(x,y):
        return array([
                        y[1],
                        -1/x*y[1]
                    ])
    #values of a,b
    xStart = a/2
    xEnd = a
    xStop = r0
    #values of y(a) and y(b)
    u0= 400/a
    u1= u0*2
    #Step size
    X,Y = shooting_o2(F_8_2_18,xStart,0,xEnd,200,u0,u1,xStop)
    return Y[-1,0]
    raise Exception("Not implemented")

'''
    Utilities for Initial Value Problems and Two-Point Boundary Problems
'''
#From course notes, slightly modified
def runge_kutta_4_modified(F, x0, y0, x, h):
    '''
   Return y(x) given the following initial value problem:
   y' = F(x, y)
   y(x0) = y0 # initial conditions
   h is the increment of x used in integration
   F = [y'[0], y'[1], ..., y'[n-1]]
   y = [y[0], y[1], ..., y[n-1]]
   '''
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while y0[0] < x:   #modification of code from course
        k0 = F(x0, y0)
        k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
        k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
        k3 = F(x0 + h, y0 + h * k2)
        y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)

#From course notes
def runge_kutta_4(F, x0, y0, x, h):
    '''
   Return y(x) given the following initial value problem:
   y' = F(x, y)
   y(x0) = y0 # initial conditions
   h is the increment of x used in integration
   F = [y'[0], y'[1], ..., y'[n-1]]
   y = [y[0], y[1], ..., y[n-1]]
   '''
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0 + h / 2.0, y0 + h / 2.0 * k0)
        k2 = F(x0 + h / 2.0, y0 + h / 2 * k1)
        k3 = F(x0 + h, y0 + h * k2)
        y0 = y0 + h / 6.0 * (k0 + 2 * k1 + 2.0 * k2 + k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)

#From course notes, slightly modified
def false_position(f, a, b, delta_x):
    '''
    f is the function for which we will find a zero
    a and b define the bracket
    delta_x is the desired accuracy
    Returns ci such that |ci-c_{i-1}| < delta_x
    '''
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        raise Exception("Root hasn't been bracketed")
    estimates = []
    while True:
        c = (a*fb-b*fa)/(fb-fa)
        estimates.append(c)
        fc = f(c)
        if sign(fc) == sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if len(estimates) >=2 and abs(estimates[-1] - estimates[-2]) <= delta_x:
            break
    return c, estimates

#From course notes, slightly modified
def shooting_o2(F, a, alpha, b, beta, u0, u1, xStop,delta=10E-3):
    '''
    Solve the boundary condition problem defined by:
    y' = F(x, y)
    y(a) = alpha
    y(b) = beta
    u0 and u1 define a bracket for y'(a)
    delta is the desired accuracy on y'(a)
    Assumes problem is of order 2 (F has two coordinates, alpha and beta are scalars)
    '''

    def r(u):
        '''
        Boundary residual, as in equation (1)
        '''
        # Estimate theta_u
        # Evaluate y and y' until x=b, using initial condition y(a)=alpha and y'(a)=u
        X, Y = runge_kutta_4(F, a, array([alpha, u]), b, 0.2)
        theta_u = Y[-1, 0]  # last row, first column (y)
        return theta_u - beta

    # Find u as a the zero of r
    u, _ = false_position(r, u0, u1, delta)

    # Now use u to solve the initial value problem one more time
    X, Y = runge_kutta_4(F, a, array([alpha, u]), xStop, 0.2)       #line modified
    return X, Y

'''
    Utilities for Interpolation and Curve Fitting
'''

#From course notes, CHAPTER 3, slightly modified
def polynomial_fit(x_data, y_data, m):
    '''
    Returns the ai
    '''
    # x_power[i] will contain sum_i x_i^k, k = 0, 2m
    x_powers = zeros(2*m+1, dtype=float_)
    b = zeros(m+1, dtype = float_)
    for i in range(2*m+1):
        x_powers[i] = sum(x_data**i)
        if i < (m+1):
            b[i] = sum(y_data*x_data**i)
    a = zeros((m+1, m+1), dtype = float_)
    for k in range(0,m+1):
        for j in range(0,m+1):
            a[k, j] = x_powers[j+k]
    return gauss_multiple_pivot(a,b)

'''
    Utilities for integration
'''

def poly_eval(coeffs,x):
    n = coeffs.size
    p = zeros(x.size)   #stores value of polynomial
    for i in range(0,n):
        p+= coeffs[i]*x**i
    return p

def gauss_leg_n2(a,b,x_data,y_data):
    #Determine evaluation points (variable transformation)
    x = zeros(3, dtype=float_)
    x[0]=(b+a)/2+(b-a)/2*(-0.77459666924)
    x[1]=(b+a)/2
    x[2]=(b+a)/2+(b-a)/2*0.77459666924

    #interpolate data points using polynomial fit
    m = len(x_data)-1
    coeffs = polynomial_fit(x_data,y_data,m) #returns coefficients of polynomial interpolant

    #Evaluate polynomial at evaluation points
    p = poly_eval(coeffs,x)    #evaluates polynomial at the data points

    #Calculate quadrature
    q = array([5/9,8/9,5/9],dtype=float_)
    I = (b-a)/2*sum((q*p))
    return I

#From course notes
def trapezoid_rec(f, a, b, tol=10E-12, max_iters=100):
    '''
    Integrates f between a and b using the recursive rule,
    until an accuracy of tol is reached.
    '''
    n = 1
    I = (b-a)/2*(f(a)+f(b)) # trapezoid with 1 panel
    bound = 1
    for k in range(2, max_iters):
        s = 0
        for i in range(1, bound+1):
            s += f(a+((2*i-1)*(b-a))/(2*bound))  # bound is 2**(k-2)
        bound *= 2  # now bound is 2**(k-1)
        J = 1/2*I + (b-a)/bound*s
        if abs(J-I) < tol:
            print("Used {} panels".format(k))
            return J
        I = J
    raise Exception("Did not converge!")




'''
    Solver from From A1
'''

def gauss_multiple_pivot(a, b):
    '''
      Task: This function returns the same result as the previous one,
            except that it uses scaled row pivoting.
      Parameters: a is a numpy array representing a square matrix. b is a numpy
            array representing a matrix with as many lines as in a.
      Test: This function is is tested by the function
            test_gauss_multiple_pivot in tests/test_gauss_multiple.py.
    '''

    ## YOUR CODE GOES HERE
    gauss_elimin_pivot(a,b)
    ''' The determinant of a triangular matrix 
        is the product of the diagonal elements
    '''
    det = prod(diagonal(a))
    assert(det!=0)
    return gauss_substitution(a,b)
    raise Exception("Function not implemented")



#for gauss_multiple_pivot
def gauss_elimin_pivot(a,b,verbose=False):
    #A
    n, m = shape(a)     #must be square
    #B
    n2=1
    m2=1
    if len(shape(b))==1:
        n2, = shape(b)   #does not need to be square
    elif len(shape(b))==2:
        n2, m2 = shape(b)   #does not need to be square
    else:
        raise Exception("B has more than 2 dimensions.")
    assert(n==n2)
    #Used for pivoting
    s = zeros(n, dtype =float_)
    for i in range (0,n):
        s[i] = max(abs(a[i, :])) #max of row i in A
    # Pivoting
    #print(a)
    for k in range (0,n-1):     #range(start,stop[,step])
        p = argmax(abs(a[k:, k]) / s[k:]) + k
        swap(a,p,k) #swap rows in matrix A
        swap(b,p,k) #swap rows in matrix b
        swap(s,p,k) #swap rows in vector  s
        #Apply row operations
        for i in range (k+1, n):
            assert(a[k,k]!=0) #verify what to do later
            if(a[i,k]!=0): #no need to do anything when lambda is 0
                lmbda = a[i,k]/a[k,k]
                a[i,k:n]=a[i,k:n] - lmbda * a[k,k:n] #apply operation to row i of A
                if m2==1:
                    b[i] = b[i] - lmbda * b[k]  # apply operation to row i of b
                else:
                    b[i,:]=b[i,:] - lmbda * b[k,:] #apply operation to row i of b
            if verbose:
                print('a:\n', a, '\nb:\n', b, '\n')


def gauss_substitution(a, b):
    n, m = shape(a)
    # Verify the n*n dimensions of B
    n2 = 1
    m2 = 1
    if len(shape(b)) == 1:
        n2, = shape(b)
    elif len(shape(b)) == 2:
        n2, m2 = shape(b)
    else:
        raise Exception("B has more than 2 dimensions")
    assert (n == n2)
    if m2 > 1:
        x = zeros([n, m2], dtype=float_)
        for i in range(n - 1, -1,
                       -1):  # decreasing index, #range(start,stop[,step]) -> iterates over every row of solution matrix x
            for j in range(0, m2):
                x[i, j] = (b[i, j] - dot(a[i, i + 1:], x[i + 1:, j])) / a[i, i]
        # return n*m system of solutions
        return x
    else:
        x = zeros([n], dtype=float_)
        for i in range(n - 1, -1,
                       -1):  # decreasing index, #range(start,stop[,step]) -> iterates over every row of solution matrix x
            x[i] = (b[i] - dot(a[i, i + 1:], x[i + 1:])) / a[i, i]
        # return n*m system of solutions
        return x

    # for gauss_multiple_pivot


def swap(a, i, j):
    if len(shape(a)) == 1:
        a[i], a[j] = a[j], a[i]  # unpacking
    else:
        a[[i, j], :] = a[[j, i], :]