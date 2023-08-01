# -*- coding: utf-8 -*-
"""TAREA 9 - FEM for Heat Eq w Crank Nicolson

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f4qmp0PMmFYEh38nLdDzbqhi-X4Jozcm

Buscamos resolver numéricamente la ecuación
\begin{align*}
  u_t  &= \alpha^2 u_{xx} \\
u(x,0) &= g(x)\\
u(0,t) &= A \\
u(L,t) &= B \\
(t,x) &\in \mathbb{R}^{+} \times (0,L)
\end{align*}
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
from numpy import exp as exp
from numpy import cos as cos
from numpy import sin as sin

solution = lambda x,t: sin(pi * x) * exp(-pi**2 * t)
g = lambda x: sin(pi *x)
alfa = 1
L,T = 1, 1 # T = límite de la discretización del tiempo
A, B = 0, 0
n,m = 10,50
xx = np.linspace(0,L,n+1)
tt = np.linspace(0,T,m+1)
dx, dt = L/n, T/m

"""Los datos previamente establecidos corresponden al problema
\begin{align*}
u_t &= u_{xx}, \quad x \in (0,1) , t>0 \\
u(x,0) &= \sin (\pi x)\\
u (0,t) &= 0\\
u (1,t) &= 0
\end{align*}

Con solución exacta $$ u(x,t) = \sin (\pi x) e^{-\pi^2 t}.$$

#Método de elementos finitos

Considerar
$$ a_{ij} = \int_0^L \phi_i \phi_j dx$$
$$ b_{ij} = \int_0^L \phi_i' \phi_j' dx$$

Si $$ u_h(x,t) = \sum_j c_j(t) \phi_j(x), $$ entonces el sistema a resolver para hallar los $c_j$ es: $$ AC' + BC = 0, $$ y al discretizar el tiempo $\{t_i\}_0^n$, y usando el esquema de Crank - Nicolson, obtenemos el sistema iterativo $$ \left(\frac{1}{dt} A + \frac{1}{2} B\right)C^{k+1} = \left(\frac{-1}{2} B +\frac{1}{dt} A\right) C^k, $$ donde $dt$ es el paso en la discretización del tiempo.

Note que la matriz $B$ ya fue obtenida en clases anteriores. Tenemos
$$ a_{ii} = 2dx/3$$
$$ a_{i,i+1} = dx/6$$
$$ b_{ii} = 2/dx$$
$$ b_{i,i+1} = -1/dx$$
$$ b_{00} = b_{nn} = 1/dx$$
$$ a_{00} = a_{nn} = dx/3$$
$a_{ij} = b_{ij} = 0$ en otro caso.
"""

'''
    Ensamblando la matriz A y B
'''

Amat = np.zeros((n+1,n+1),float)
Bmat = np.zeros((n+1,n+1),float)

np.fill_diagonal(Amat, np.full(n+1, 2*dx/3))
np.fill_diagonal(Amat[1:], np.full(n,dx/6))
np.fill_diagonal(Amat[:, 1:], np.full(n,dx/6))
np.fill_diagonal(Bmat, np.full(n+1,2/dx))
np.fill_diagonal(Bmat[1:], np.full(n,-1/dx))
np.fill_diagonal(Bmat[:, 1:], np.full(n,-1/dx))
Bmat = alfa**2 * Bmat

Amat_for_solve = Amat/dt + Bmat/2
Amat_for_solve[0,0] = 1
Amat_for_solve[0,1] = 0
Amat_for_solve[1,0] = 0
Amat_for_solve[-1,-1] = 1
Amat_for_solve[-2,-1] = 0
Amat_for_solve[-1,-2] = 0

# np.savetxt("Matriz A", Amat, fmt = '%.2e')
# np.savetxt("Matriz B", Bmat, fmt = '%.2e')

"""Ahora analicemos los elementos $c_k(t_l)$.

Note que para hallar $c_i(0)$, tenemos $$ g(x) = u(x,0) = \sum_j c_j(0) \phi_j (x). $$ Así $$ A = u(0,t) = \sum_j c_j(t) \phi_j(0) = \sum_j c_j(t) \phi_j(x_0) = \sum_j c_j(t) \delta_{j0} = c_0(t)$$ para todo $t$, así $c_0(t_l) = A$ para todo $l = 0,..., m$ y análogamente $c_m(t_l) = B$.

Si almacenamos una matriz $C = (C^0(tt), C^1(tt), ... , C^m(tt))$, entonces tenemos conocida la primera fila, la última fila. $$ C_{kl} = c_k(t_l)$$

Para resolver el primer vector $C^0$, note que $$AC^0 = \int_0^L g\phi $$
"""

from functools import partial
from scipy.special import roots_legendre

h = dx
def gauss1d(fun, x0, x1):
    n = 5
    xi, wi = roots_legendre(n)
    inte = 0
    h = 0.5 * (x1 - x0)
    xm = 0.5 * (x0 + x1)
    for cont in range(n):
        inte += h * fun(h * xi[cont] + xm) * wi[cont]
    return inte

phi0_e = lambda x: 1-x/h
phi1_e = lambda x: x/h
ftimesphi0_e = lambda x,i: g(x+xx[i])*phi0_e(x)
ftimesphi1_e = lambda x,i: g(x+xx[i])*phi1_e(x)
b_right_for_c0 = np.zeros(n+1,float)
for i in range(n):
  aux0 = partial(ftimesphi0_e, i = i)
  aux1 = partial(ftimesphi1_e, i = i)
  b_right_for_c0[i]+= gauss1d(aux0, 0, h)
  b_right_for_c0[i+1]=gauss1d(aux1, 0, h)

Csol = np.zeros((n+1,m+1),float)

r = -A*Amat[:,0]-B*Amat[:,-1]

# print(Amat_for_solve)

b_right_for_c0 += r
b_right_for_c0[0]  = A
b_right_for_c0[-1] = B

# Csol[:,0] = np.linalg.solve(Amat_for_solve,
                            # b_right_for_c0)
Csol[:,0] = g(xx)
for k in range(1,m+1):
  bsolv = np.dot((Amat/dt - Bmat/2),Csol[:,k-1])+r
  bsolv[0]  = A
  bsolv[-1] = B
  Csol[:,k] = np.linalg.solve(Amat_for_solve, bsolv)

"""#Ploteando x"""

xfixed = int(n/2)
if xfixed <= n:
  plt.plot(tt,solution(xx[xfixed],tt), c = 'blue', label = 'solucion en x =' + str(xx[xfixed]))
  plt.scatter(tt,Csol[xfixed,:], s = 2, c = 'red', label= 'discretizacion en x=' + str(xx[xfixed]))
  plt.title("Solución exacta $u(x,t) = \sin (\pi x) e^{-\pi^2 t}$")
  plt.legend()
else:
  print("error")

# np.savetxt("Csol", Csol, fmt = '%2.2f')

"""#Ploteando t"""

tfixed = int(m/2)
if tfixed <= m:
  plt.plot(xx,solution(xx,tt[tfixed]), c = 'blue', label = 'solucion en x =' + str(tt[tfixed]))
  plt.scatter(xx,Csol[:,tfixed], s = 10, c = 'red', label= 'discretizacion en x=' + str(tt[tfixed]))
  plt.title("Solución exacta $u(x,t) = \sin (\pi x) e^{-\pi^2 t}$")
  plt.legend()

else:
  print("error")