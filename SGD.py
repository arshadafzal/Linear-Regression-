#  Program to find Regression Co-efficients in Linear Regression using the Stochastic Gradient-Descent Method.
#  The objective function  to be minimized is the Error function, (1/2)*(Y-X*Theta)^2.
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
#  Maximum number of iterations,set it to large value
#  Relative error tolerance for Theta
#  Learning rate, Alpha
#  =========
#  OUTPUT:
#  =========
#  Least-square estimate of 'Theta'
#  =================================================================================
#  Tip: The value of alpha, will affect the solution, and must be selected carefully
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
#  Parameters for optimization
n = int(input('Enter the number of Data-points :'))
maxiter = int(input('Enter the maximum number of iterations :'))
tol = float(input('Enter the tolerance for variable theta :'))
alpha = float(input('Enter the learning rate for gradient descent algorithm :'))
#  Initialization
showiterinfo = True
theta = np.random.uniform(0, 1, [q - 1, 1])
t = theta
#  Main Loop
for i in range(maxiter):
    # Loop with respect to Each Data Points
    for j in range(n):
        # Derivative(Finding the residual)
        r = np.dot(np.transpose(x[j:j+1, 0:q-1]), (np.dot(x[j:j+1, 0:q-1], theta) - y[j]))
        # Updating Theta
        theta = theta - alpha * r
    # Mean-squared Error with respect to iteration counter
    f_value = (1 / (2 * n)) * np.sum(pow((np.dot(x, theta) - y), 2), axis=0)
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
# Print output
print(theta)

