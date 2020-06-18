#  Program to find Regression Co-efficients in Linear Regression using the Quasi- Newton Methods
#  The objective function  to be minimized is the Error function, (1/2*N)*(Y-X*Theta)^2.
#  Here, N is the number of data-points. 'X' is the matrix of Predictors/features (N-by-D);
#  Columns of X may also include terms obtained through non-linear transformation
#  Example: using 2 variables x1 and x2, the columns of X can be 1, x1, x2, x1^2, x1*x2 etc.
#  'Y' is vector of Response (N-by-1)
#  'Theta' is the vector of co-efficients to be determined (D-by-1)
#  =========
#  INPUTS:
#  =========
#  'X' Matrix of Predictors/features (N-by-D)
#  'Y' Vector of Response (N-by-1)
#  =========
#  OUTPUT:
#  =========
#  Least-square estimate of 'Theta'
#  =================================================================================
#  Version 1.0
#  Author: Arshad Afzal, IIT Kanpur, India
#  https://www.researchgate.net/profile/Arshad_Afzal
#  For Questions/ Comments, please email to arshad.afzal@gmail.com
#  =================================================================================
import xlrd
import numpy as np
#  Exporting data from excel workbook
wb = xlrd.open_workbook('Data.xlsx')
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
p = sheet.nrows
q = sheet.ncols
x = np.zeros([p, q-1])
y = np.zeros([p, 1])
for i in range(sheet.nrows):
    for j in range(sheet.ncols - 1):
        x[i][j] = sheet.cell_value(i, j)
    y[i] = sheet.cell_value(i, sheet.ncols - 1)
n = int(input('Enter the number of Data-points :'))
maxiter = int(input('Enter the maximum number of iterations :'))
tol = float(input('Enter the tolerance for variable theta :'))
option = (input('Enter the option as DFP or BFGS :'))
#  Initialization
showiterinfo = True
theta = np.random.uniform(0, 1, [q - 1, 1])
b = np.identity(q - 1)
t = theta
for i in range(maxiter):
    #  Derivative of Function
    r = np.dot(np.transpose(x), (np.dot(x, t) - y))
    #  Direction Vector
    d = np.matmul(b, r)
    # Learning Parameter Alpha
    alpha = np.dot(np.transpose(d), r) / np.dot(np.transpose(d), (np.dot((np.matmul(np.transpose(x), x)), d)))
    #  One-step Learning of Theta
    theta = theta - alpha * d
    # Mean-squared Error
    f_value = (1 / (2 * n)) * np.sum(pow((np.dot(x, theta) - y), 2), axis=0)
    # Find new approximation of the Hessian, B
    delta = theta - t
    gamma = np.dot(np.transpose(x), (np.dot(x, theta) - y)) - r
    if option == 'DFP':
        #  Finding B using DFP Updates
        p = np.dot(delta, np.transpose(delta))/np.dot(np.transpose(delta), gamma)
        num = np.dot(b, np.dot(gamma, (np.dot(np.transpose(gamma), b))))
        den = np.dot(np.transpose(gamma), (np.dot(b, gamma)))
        q = num/den
        b = b + p - q
    else:
        #  Finding B using BFGS Updates
        scalar = 1 + (np.dot(np.transpose(gamma), (np.dot(b, gamma))) / np.dot(np.transpose(delta), gamma))
        pp = scalar * np.dot(delta, np.transpose(delta)) / np.dot(np.transpose(delta), gamma)
        numm = np.dot(delta, np.dot(np.transpose(gamma), b)) + np.dot(b, np.dot(gamma, np.transpose(delta)))
        denm = np.dot(np.transpose(delta), gamma)
        qq = numm / denm
        b = b + pp - qq
        # Relative change in Theta using squared norm
    err_theta = np.linalg.norm(theta - t) / np.linalg.norm(theta)
    # Display iteration info
    if showiterinfo:
        print("Iteration " + str(i) + ":  " + "Function value: " + str(f_value))
    # Check for convergence
    if err_theta <= tol:
        print("\nChange in theta less tha specified tolerance\n")
        break
    t = theta
#  Print Output
print(theta)

