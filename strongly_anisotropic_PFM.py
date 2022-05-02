# ===================================
# FEniCS code  Variational Fracture Mechanics
# ===================================
#
# A quasi-static solution of the variational fracture mechanics problems
# using the regularization strongly anisotropic damage model
#
# author: Bin Li (bin.l@gtiit.edu.cn)
#
# The code use the MITC elements from fenics-shells
#
# date: 07/16/2021
# ----------------
# References:
# Ref1: He, Q-C., and Q. Shao. "Closed-form coordinate-free decompositions of the 
#       two-dimensional strain and stress for modeling tension–compression 
#       dissymmetry." Journal of Applied Mechanics 86.3 (2019): 031007.


from __future__ import division
import sys
from dolfin import *
from mshr import *
from ufl import RestrictedElement, ne

import fem
from fem.utils import inner_e
from fem.functionspace import *

#import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
sys.path.append("../")
from utils import save_timings
import matplotlib.pyplot as plt
import petsc4py
petsc4py.init(sys.argv)
# -----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER
# -----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)  # log level
# set some dolfin specific parameters
parameters["use_petsc_signal_handler"] = True
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 2
# -----------------------------------------------------------------------------
# parameters of the solvers
"""
solver_u_parameters = {"nonlinear_solver": "newton",
                       "newton_solver": {"linear_solver": "mumps",
                                         #"preconditioner" : "hypre_amg",
                                         "maximum_iterations": 40,
                                         "absolute_tolerance": 1e-8,
                                         "relative_tolerance": 1e-8,
                                         "report": True,
                                         "error_on_nonconvergence": False}} 
"""
solver_u_parameters  = {"nonlinear_solver": "snes",
                        "symmetric": True,
                        "snes_solver": {"linear_solver": "mumps", #"mumps",
                                        "method" : "newtontr", #"newtontr", "newtonls"
                                        "line_search": "cp", # "basic", "bt", "l2", "cp", "nleqerr"
                                        "preconditioner" : "hypre_amg", 
                                        "maximum_iterations": 60,
                                        "absolute_tolerance": 1e-10,
                                        "relative_tolerance": 1e-10,
                                        "solution_tolerance": 1e-10,
                                        "report": True,
                                        "error_on_nonconvergence": False}}  

#PETScOptions.set("help")
PETScOptions.set("snes_type","vinewtonssls")
PETScOptions.set("snes_converged_reason")
PETScOptions.set("snes_linesearch_type","basic") #shell basic l2 bt nleqerr cp
PETScOptions.set("ksp_type","preonly")
PETScOptions.set("pc_type","lu")
PETScOptions.set("pc_factor_mat_solver_type","mumps")
#PETScOptions.set("snes_report")
#PETScOptions.set("snes_monitor")
PETScOptions.set("snes_vi_zero_tolerance",1.e-6)
PETScOptions.set("snes_stol",1.e-6)
PETScOptions.set("snes_atol",1.e-6)
PETScOptions.set("snes_rtol",1.e-6)
PETScOptions.set("snes_max_it",800)
PETScOptions.set("snes_error_if_not_converged",False)
PETScOptions.set("snes_force_iteration",1)

#----------------------------------------------------------------------------
# set the user parameters
userpar = Parameters("user")
#----------------------------------------------------------------------------
#E  = 1.0e3
#nu = 0.3

#mu = E/(2.0*(1.0+nu))
#lmbda = E*nu/(1.0-2.0*nu)/(1.0+nu)  # plane strain
# isotropic elasticity
#userpar.add("L1111", 2.*mu+lmbda)
#userpar.add("L2222", 2.*mu+lmbda)
#userpar.add("L1122", lmbda)
#userpar.add("L1212", mu)
# isotropic surface energy
# userpar.add("C11",1.0)
# userpar.add("C22",1.0)
# userpar.add("C12",0.5)
# userpar.add("C44",0.25)
#----------------------------------------------------------------------------


# anisotropic elasticity taking from Nguyen2020-IJNME
# userpar.add("L1111", 6.5e4)
# userpar.add("L2222", 2.6e5)
# userpar.add("L1122", 2.0e4)
# userpar.add("L1212", 3.0e4)
#----------------------------------------------------------------------------
# userpar.add("L1111", 13.1411e3)
# userpar.add("L2222", 65.7053e3)
# userpar.add("L1122", 6.9779e3)
# userpar.add("L1212", 2.425e3)
#----------------------------------------------------------------------------
userpar.add("L1111", 1.27e5)
userpar.add("L2222", 1.27e5)
userpar.add("L1122", 7.08e4)
userpar.add("L1212", 7.355e4)
# Isotropic Elasticity
# userpar.add("L1111", 1.27e5)
# userpar.add("L2222", 1.27e5)
# userpar.add("L1122", 7.08e4)
# userpar.add("L1212", 2.81e4)
#----------------------------------------------------------------------------
# anisotropic phase-field
userpar.add("C11",1.8)
userpar.add("C22",1.8)
userpar.add("C12",-1.7)
userpar.add("C44",0.15)
#----------------------------------------------------------------------------
userpar.add("Gc",1.0)
userpar.add("ell",6.0*0.003)
userpar.add("k_ell",1.0e-6)
userpar.add("KI",1.) # mode I loading
userpar.add("KII",0.) # mode II loading
userpar.add("K0",2.0e3)
userpar.add("theta0",2.0)
# userpar.add("theta0",0.0) # for elastic stiffness
# userpar.add("theta00",0.0) # for surface energy stiffness
# userpar.add("load_min",2.9e-3)
userpar.add("load_min",0.)
userpar.add("load_max",0.0050)
userpar.add("load_steps",501)
userpar.add("project", True)
parameters.add(userpar)
parameters.parse()
info(parameters,True)
userpar = parameters["user"]


if userpar["project"] == True:
    userpar.add("MITC","project") 
    if MPI.rank(MPI.comm_world) == 0:
    	print("Solving the damage sub-problem in the PROJECTED SPACE \nwith static condensation of the local variable\n")
else:
    userpar.add("MITC","full") 
    if MPI.rank(MPI.comm_world) == 0:
    	print("Solving the damage sub- problem in the FULL SPACE")

theta0 = userpar["theta0"]*np.pi/180.0
# theta00 = userpar["theta00"]*np.pi/180.0
Gc = userpar["Gc"]
ell= userpar["ell"]
# residual stiffness
k_ell = userpar["k_ell"]
KI = userpar["KI"]
KII = userpar["KII"]
K0 = userpar["K0"]


# ----------------------------------------------------------------------------
# Parameters for surface energy and materials
# ----------------------------------------------------------------------------

# Material constant
# -----------------------------------------------------------------------------
# stiffness matrix in Voigt notation
L4 = np.matrix([[userpar["L1111"], userpar["L1122"], 0.0],\
                [userpar["L1122"], userpar["L2222"], 0.0],\
                [0.0, 0.0, userpar["L1212"]]]) 

L2 = np.matrix([[L4[0,0], L4[0,1], sqrt(2.0)*L4[0,2]],\
                [L4[1,0], L4[1,1], sqrt(2.0)*L4[1,2]],\
                [sqrt(2.0)*L4[2,0], sqrt(2.0)*L4[2,1], 2.0*L4[2,2]]]) 

# Constitutive matrix Cmat for the fourth order phase-field and its rotated matrix Cmatr
Cmat = np.matrix([[userpar["C11"], userpar["C12"], 0],\
                  [userpar["C12"], userpar["C22"], 0],\
                  [0, 0, userpar["C44"]]])

# -----------------------------------------------------------------------------
# rotational matrix for 4th-order tensor in Voigt notation
K = [[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)], \
     [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)], \
     [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]]
# K1 = [[np.cos(theta00)**2, np.sin(theta00)**2 ,  2.0*np.cos(theta00)*np.sin(theta00)], \
#      [np.sin(theta00)**2, np.cos(theta00)**2 , -2.0*np.cos(theta00)*np.sin(theta00)], \
#      [-np.cos(theta00)*np.sin(theta00), np.cos(theta00)*np.sin(theta00) , np.cos(theta00)**2-np.sin(theta00)**2]]
# -----------------------------------------------------------------------------
# rotation matrix for 3D second-order stiffness matrix
R = 0.5*np.matrix([[1+np.cos(2*theta0), 1-np.cos(2*theta0),  np.sqrt(2)*np.sin(2*theta0)],\
                   [1-np.cos(2*theta0), 1+np.cos(2*theta0), -np.sqrt(2)*np.sin(2*theta0)],\
                   [-np.sqrt(2)*np.sin(2*theta0), np.sqrt(2)*np.sin(2*theta0), 2*np.cos(2*theta0)]])

def npmat2matrix(Cmatr):
    # convert np.matrix to as_matrix format
    Cr11 = Cmatr[0,0]
    Cr12 = Cmatr[0,1]
    Cr13 = Cmatr[0,2]
    Cr22 = Cmatr[1,1]
    Cr23 = Cmatr[1,2]
    Cr33 = Cmatr[2,2]
    return as_matrix([[Cr11, Cr12, Cr13], [Cr12, Cr22, Cr23], [Cr13, Cr23, Cr33]])

# -----------------------------------------------------------------------------
L4matr = np.matmul(np.matmul(K,L4), np.transpose(K))
L4r = npmat2matrix(L4matr)

Cmatr = np.matmul(np.matmul(K,Cmat), np.transpose(K))
Crv = npmat2matrix(Cmatr)


# Loading
ut = 1.0   # reference value for the loading (imposed displacement)

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4

# Geometry paramaters
H = 2.0
H1 = 2.0
H2 = 0.0
# H1 = H/2
# H2 = -H/2
hsize = 0.003

modelname = "aniso_trapezoidal_specimen"
meshname  = modelname+"-mesh.xdmf"
simulation_params = "Aniso_elastic_strapezoidal_specimen_C11_%.4f_C22_%.4f__C12_%.4f_C44_%.4f_theta0_%.4f_ell_%.4f_K0_%.1e_%s" %(userpar["C11"],\
                     userpar["C22"], userpar["C12"], userpar["C44"], userpar["theta0"], ell, K0, userpar["MITC"])

savedir = "output/"+modelname+"/"+simulation_params+"/"

if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

# read mesh
mesh = Mesh("../meshes/trapezoidal_specimen.xml")

geo_mesh  = XDMFFile(MPI.comm_world, savedir+meshname)
geo_mesh.write(mesh)

mesh.init()
ndim = mesh.geometry().dim() # get number of space dimensions
if MPI.rank(MPI.comm_world) == 0:
    print("the dimension of mesh: {0:2d}".format(ndim))

# -----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
# -----------------------------------------------------------------------------
# Strain and stress
def eps(v):
    return sym(grad(v))

def strain2voigt(e):
    return as_vector([e[0,0],e[1,1],2.0*e[0,1]])

def voigt2strain(e):
    return as_tensor([[e[0], 0.5*e[2]],
                      [0.5*e[2], e[1]]])
def voigt2stress(s):
    return as_tensor([[s[0], s[2]],
                      [s[2], s[1]]])

def sigma_0(v):
    # Plane Strain
    return voigt2stress(dot(L4r, strain2voigt(eps(v)))) 

# Constitutive functions of the damage model
def w(alpha):
    return 9.0*alpha

def aa(alpha):
    return (1-alpha)**2


# --------------------------------------------------------------------
# Dirichlet boundary condition
# --------------------------------------------------------------------

# =============================================================================
def boundaries(x):
    return near(x[1], H1, 0.1*hsize) or near(x[1], H2, 0.1*hsize) \
        or near(x[0], 0.0, 0.1*hsize) or near(x[0], H, 0.1*hsize)
# =============================================================================

# =============================================================================
# upper = CompiledSubDomain("near(x[1], %s)"%H1)
# lower = CompiledSubDomain("near(x[1], %s)"%H2)
#
# boundaries1 = MeshFunction("size_t", mesh,1)
# boundaries1.set_all(0)
# upper.mark(boundaries1, 1)
# lower.mark(boundaries1, 2)
# # left_upper: ds(1), right: ds(2)
# ds = Measure("ds",subdomain_data=boundaries1)
# =============================================================================
## when using "pointwise", the boolean argument on_boundary
## in SubDomain::inside will always be false

def left_pinpoints(x, on_boundary):
    return  near(x[0], -4.0, 0.2 * hsize) and near(x[1], 0.0, 0.2 * hsize)
def right_pinpoints(x, on_boundary):
    return  near(x[0], 4.0, 0.2 * hsize) and near(x[1], 0.0, 0.2 * hsize)
def upper_pinpoints(x, on_boundary):
    return  near(x[0], 0.0, 0.2 * hsize) and near(x[1], 2.0, 0.2 * hsize)

# =============================================================================

def upper_boundary(x, on_boundary):
    return on_boundary and near(x[1],  0.16*x[0]+0.2, 0.1 * hsize)

def lower_boundary(x, on_boundary):
    return on_boundary and near(x[1], -0.16*x[0]-0.2, 0.1 * hsize)
# =============================================================================

# =============================================================================  
# -----------------------------------------------------------------------------
# Variational formulation
# -----------------------------------------------------------------------------
# Create function space for 2D elasticity
V_u = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create the function space for the damage using mixed formulation
# see fenics-shells for further details
element_alpha = FiniteElement("Lagrange", triangle, 1)
element_a = VectorElement("Lagrange", triangle, 2)
element_s = FiniteElement("N1curl", triangle, 1)
element_p = RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")
element = MixedElement([element_alpha,element_a,element_s,element_p])
V_alpha = FunctionSpace(mesh,element_alpha)
V_a = FunctionSpace(mesh,element_a)
V_s = FunctionSpace(mesh,element_s)
V_p = FunctionSpace(mesh,element_p)
V_damage = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
V_damage_F = V_damage.full_space
V_damage_P = V_damage.projected_space
assigner_F = FunctionAssigner(V_damage_F,[V_alpha,V_a,V_s,V_p])
assigner_P = FunctionAssigner(V_damage_P,[V_alpha,V_a])

# Define the function, test and trial fields
u  = Function(V_u, name="Displacement")
du = TrialFunction(V_u)
v  = TestFunction(V_u)
damage = Function(V_damage_F, name="Damage")
damage_trial = TrialFunction(V_damage_F)
damage_test = TestFunction(V_damage_F),
alpha, a, s, p = split(damage)
damage_p = Function(V_damage_P, name="Damage")

# Define the bounds for the damage field
alpha_ub = Function(V_alpha)
alpha_lb = Function(V_alpha)
a_lb = Function(V_a)
a_ub = Function(V_a)
s_lb = Function(V_s)
s_ub = Function(V_s)  
p_lb = Function(V_p)
p_ub = Function(V_p)
alpha_ub.vector()[:] = 1.
alpha_lb.vector()[:] = 0.

#########
# alpha_lb = interpolate(Expression("x[0]<=0.5 & near(x[1], 0.0, tol) ? 1.0 : 0.0", \
#                        tol = 0.1*ell, degree=0), V_alpha)

# alpha_lb = interpolate(Expression("(abs(x[1])< 0.0016 + tol & x[0] < 0.5-R0) or x[1]*x[1] + (x[0]-x0)*(x[0]-x0) < R0*R0  ? 1.0 : 0.0", \
#                        tol = 0.1*0.00064, R0 = 0.0016, x0 = 0.50, degree=0), V_alpha)

for ub in [a_ub,s_ub,p_ub]:

    ub.vector()[:] = np.infty
for lb in [a_lb,s_lb,p_lb]:
    lb.vector()[:] = -np.infty

if userpar["MITC"] == "project":
    damage_lb = Function(V_damage_P); 
    damage_ub = Function(V_damage_P)
    assigner_P.assign(damage_ub,[alpha_ub, a_ub])
    assigner_P.assign(damage_lb,[alpha_lb, a_lb])
else:
    damage_lb = Function(V_damage_F); 
    damage_ub = Function(V_damage_F)
    assigner_F.assign(damage_ub,[alpha_ub,a_ub,s_ub,p_ub])
    assigner_F.assign(damage_lb,[alpha_lb,a_lb,s_lb,p_lb])
    
# =============================================================================  

# Dirichlet boundary conditions
# bc - u (imposed displacement)


# =============================================================================
# u_R1 = Expression(("0.0","t"), t = 0.0, degree=0)
# bcu_1 = DirichletBC(V_u, u_R1, boundaries1, 1)
# bcu_2 = DirichletBC(V_u, ("0", "0"), boundaries1, 2)
# bc_u = [bcu_1, bcu_2]
# =============================================================================

# =============================================================================
# u_R1 = Expression("-t", t = 0.0, degree=0)
# bcu_1 = DirichletBC(V_u, ("0", "0"), left_pinpoints, method='pointwise')
# bcu_2 = DirichletBC(V_u.sub(1), "0", right_pinpoints, method='pointwise')
# bcu_3 = DirichletBC(V_u.sub(1), u_R1, upper_pinpoints, method='pointwise')
# bc_u = [bcu_1, bcu_2, bcu_3]
# =============================================================================

# =============================================================================
# Dirichlet boundary condition
u_UL = Expression(["0.0","0.0"], degree=0)
u_UU = Expression(["0.0","t"], t=0.0, degree=0)
bc_u = [DirichletBC(V_u, u_UL, lower_boundary),\
	#DirichletBC(V_u.sub(0), u_UL, lower_pinpoints, method='pointwise'),\
	DirichletBC(V_u, u_UU, upper_boundary)]

if userpar["MITC"] == "project":
    bc_damage = [DirichletBC(V_damage_P.sub(0), 0.0, lower_boundary),\
                 DirichletBC(V_damage_P.sub(0), 0.0, upper_boundary)]
else:
    bc_damage = [DirichletBC(V_damage_F.sub(0), 0.0, lower_boundary),\
                 DirichletBC(V_damage_F.sub(0), 0.0, upper_boundary)]
# =============================================================================

# =============================================================================
# E = 120.0e3
# nu = 0.3
# mu = float(E/(2.0*(1.0 + nu)))
# kappav = float((3.0-4.0*nu))
# nKI = float(sqrt(E*Gc))
# u_U = Expression(["t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2) + \
#                    t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0+kappa+cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2)",
#                   "t*KI*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1], x[0]-lc)))*sin(atan2(x[1], x[0]-lc)/2) + \
#                    t*KII*nKI/(2*mu)*sqrt(sqrt((x[0]-lc)*(x[0]-lc)+x[1]*x[1])/(2*pi))*(2.0-kappa-cos(atan2(x[1], x[0]-lc)))*cos(atan2(x[1], x[0]-lc)/2)"],
#                     degree=2, mu=mu, kappa=kappav, nKI=nKI, KI=KI, KII=KII, lc=H1, t=0.0)
#
# # Boundary conditions
# bc_u = [DirichletBC(V_u, u_U, boundaries)]
# =============================================================================

# bc - alpha (unbreakeble)

# if userpar["MITC"] == "project":
#     bc_damage = [DirichletBC(V_damage_P.sub(0), 0.0, boundaries1, 1),\
#                  DirichletBC(V_damage_P.sub(0), 0.0, boundaries1, 2)]
# else:
#     bc_damage = [DirichletBC(V_damage_F.sub(0), 0.0, boundaries1, 1),\
#                  DirichletBC(V_damage_F.sub(0), 0.0, boundaries1, 2)]

# if userpar["MITC"] == "project":
#     bc_damage = []
# else:
#     bc_damage = []

#--------------------------------------------------------------------
# Define the variational problem
#--------------------------------------------------------------------
# Displacement subproblem
#--------------------------------------------------------------------
# eigenvalues of 3D second-order stiffness matrix 
lmbda1 = 0.5*(L2[0,0]+L2[1,1]+sqrt((L2[0,0]-L2[1,1])**2+4.*L2[0,1]**2))
lmbda2 = 0.5*(L2[0,0]+L2[1,1]-sqrt((L2[0,0]-L2[1,1])**2+4.*L2[0,1]**2))
lmbda3 = L2[2,2]

# eigenprojectors for L2
A10 = np.matmul(L2-lmbda2*np.identity(3),L2-lmbda3*np.identity(3))/(lmbda1-lmbda2)/(lmbda1-lmbda3)
 
A30 = np.matrix([[0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0], \
                 [0.0, 0.0, 1.0]])

A20 = np.identity(3)-A10-A30 

A11 = np.matmul(np.matmul(R,A10), np.transpose(R))

A21 = np.matmul(np.matmul(R,A20), np.transpose(R))

A31 = np.matmul(np.matmul(R,A30), np.transpose(R))

pL2s0 = sqrt(lmbda1)*A11 + sqrt(lmbda2)*A21 + sqrt(lmbda3)*A31 

mL2s0 = 1.0/sqrt(lmbda1)*A11 + 1.0/sqrt(lmbda2)*A21 + 1.0/sqrt(lmbda3)*A31 

pL2s = npmat2matrix(pL2s0)
mL2s = npmat2matrix(mL2s0)

"""
L2r = np.matrix([[L4matr[0,0], L4matr[0,1], sqrt(2.0)*L4matr[0,2]],\
                 [L4matr[1,0], L4matr[1,1], sqrt(2.0)*L4matr[1,2]],\
                 [sqrt(2.0)*L4matr[2,0], sqrt(2.0)*L4matr[2,1], 2.0*L4matr[2,2]]]) 

if MPI.rank(MPI.comm_world) == 0:
    print("computed as the Eq.(34) of Ref1 L2r\n", L2r)
    print("computed as the square of pL2s\n", npmat2matrix(pL2s0*pL2s0))
"""

def strain2vector3D(e):
    return as_vector([e[0,0],e[1,1],sqrt(2.0)*e[0,1]])

def strain2new(pL2s,u):
    eps_t = dot(pL2s, strain2vector3D(eps(u)))
    return as_tensor([[eps_t[0], eps_t[2]/sqrt(2.0)],
                      [eps_t[2]/sqrt(2.0), eps_t[1]]])

def strain2origin(mL2s, strain_new):
    eps_t = dot(mL2s, strain2vector3D(strain_new))
    return as_tensor([[eps_t[0], eps_t[2]/sqrt(2.0)],
                      [eps_t[2]/sqrt(2.0), eps_t[1]]])

def eig_val1(pL2s,u):
    I1 = tr(strain2new(pL2s,u))
    I2 = det(strain2new(pL2s,u))
    return 0.5*(I1+sqrt(I1**2-4.*I2))

def eig_val2(pL2s,u):
    I1 = tr(strain2new(pL2s,u))
    I2 = det(strain2new(pL2s,u))
    return 0.5*(I1-sqrt(I1**2-4.*I2))

def eig_val1_plus(pL2s,u):
    return 0.5*(eig_val1(pL2s,u)+abs(eig_val1(pL2s,u)))
     
def eig_val2_plus(pL2s,u):
    return 0.5*(eig_val2(pL2s,u)+abs(eig_val2(pL2s,u)))

def eig_val1_minus(pL2s,u):
    return 0.5*(eig_val1(pL2s,u)-abs(eig_val1(pL2s,u)))

def eig_val2_minus(pL2s,u):
    return 0.5*(eig_val2(pL2s,u)-abs(eig_val2(pL2s,u)))

def eig_vec1(pL2s,u):
    return conditional(ne(eig_val1(pL2s,u)-eig_val2(pL2s,u), 0.), (strain2new(pL2s,u)-\
                       eig_val2(pL2s,u)*Identity(2))/(eig_val1(pL2s,u)-eig_val2(pL2s,u)),\
                       as_tensor([[1, 0],[0, 0]]))

def eig_vec2(pL2s,u):    
    return conditional(ne(eig_val1(pL2s,u)-eig_val2(pL2s,u), 0.), (eig_val1(pL2s,u)*Identity(2)-\
                       strain2new(pL2s,u))/(eig_val1(pL2s,u)-eig_val2(pL2s,u)), as_tensor([[0, 0],[0, 1]]))

def proj_tensor_bases(eigvec):
    return as_matrix([[eigvec[0,0]*eigvec[0,0],eigvec[0,0]*eigvec[1,1],eigvec[0,0]*eigvec[0,1]],\
                      [eigvec[1,1]*eigvec[0,0],eigvec[1,1]*eigvec[1,1],eigvec[1,1]*eigvec[0,1]],\
                      [eigvec[0,1]*eigvec[0,0],eigvec[0,1]*eigvec[1,1],eigvec[0,1]*eigvec[0,1]]])

def proj_tensor_plus(pL2s,u_0): 
    return conditional(ge(eig_val1(pL2s,u_0), 0.),1.,0.)*proj_tensor_bases(eig_vec1(pL2s,u_0))+\
           conditional(ge(eig_val2(pL2s,u_0), 0.),1.,0.)*proj_tensor_bases(eig_vec2(pL2s,u_0))

def proj_tensor_minus(pL2s,u_0):
    return conditional(le(eig_val1(pL2s,u_0), 0.),1.,0.)*proj_tensor_bases(eig_vec1(pL2s,u_0))+\
           conditional(le(eig_val2(pL2s,u_0), 0.),1.,0.)*proj_tensor_bases(eig_vec2(pL2s,u_0))

def strain_new_plus(pL2s,u):
    return voigt2stress(dot(proj_tensor_plus(pL2s,u),strain2voigt(strain2new(pL2s,u))))

def strain_new_minus(pL2s,u):
    return voigt2stress(dot(proj_tensor_minus(pL2s,u),strain2voigt(strain2new(pL2s,u))))

def strain_origin_plus(mL2s,pL2s,u):
    return strain2voigt(strain2origin(mL2s,strain_new_plus(pL2s,u)))

def strain_origin_minus(mL2s,pL2s,u):
    return strain2voigt(strain2origin(mL2s,strain_new_minus(pL2s,u)))

def sigma_deCom_plus(L4r,mL2s,pL2s,u):
    return voigt2stress(dot(L4r,strain_origin_plus(mL2s,pL2s,u)))

def sigma_deCom_minus(L4r,mL2s,pL2s,u):
    return voigt2stress(dot(L4r,strain_origin_minus(mL2s,pL2s,u)))

def energy_active(L4r,mL2s,pL2s,u):
    return 0.5*inner(sigma_deCom_plus(L4r,mL2s,pL2s,u),voigt2strain(strain_origin_plus(mL2s,pL2s,u)))

def energy_passive(L4r,mL2s,pL2s,u):
    return 0.5*inner(sigma_deCom_minus(L4r,mL2s,pL2s,u),voigt2strain(strain_origin_minus(mL2s,pL2s,u)))

def sigma_deCom(L4r,mL2s,pL2s,u,alpha):
    return (aa(alpha)+k_ell)*sigma_deCom_plus(L4r,mL2s,pL2s,u)+sigma_deCom_minus(L4r,mL2s,pL2s,u)

elastic_energy = ((aa(alpha)+k_ell)*energy_active(L4r,mL2s,pL2s,u)+energy_passive(L4r,mL2s,pL2s,u))*dx

def sigma(u, alpha):
    return (aa(alpha)+k_ell)*sigma_0(u)

elastic_energy_new = 0.5*inner(sigma(u, alpha), eps(u))*dx

body_force = Constant((0.,0.))
external_work = dot(body_force, u)*dx
# elastic_potential = elastic_energy - external_work
elastic_potential = elastic_energy - external_work
# Weak form of elasticity problem
E_u  = derivative(elastic_potential,u,v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = derivative(E_u, u, du)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(E_u, u, bc_u, J=E_du)
# Set up the solvers                                        
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)

#--------------------------------------------------------------------
# Damage subproblem
#--------------------------------------------------------------------
kappa_tensor = sym(grad(a)) # Hessian matrix of damage field
kappa = as_vector([kappa_tensor[0,0],kappa_tensor[1,1],2.0*kappa_tensor[0,1]])
dissipated_energy = Constant(5.0/96.0)*Gc*(w(alpha)/ell+ell**3*dot(kappa, Crv*kappa))*dx
penalty_energy = 0.5*K0*Gc*inner(s,s)*dx

# Here we show another way to apply the Duran-Liberman reduction operator,
# through constructing a Lagrangian term L_R.
# -----------------------------------------------------------------------------
# Impose the constraint that s=(grad(w)-theta) in a weak form
constraint = inner_e(grad(alpha)-a-s, p, False)
damage_functional = elastic_energy+dissipated_energy+penalty_energy+constraint
# Compute directional derivative about alpha in the test direction (Gradient)
F = derivative(damage_functional, damage, damage_test)
# Compute directional derivative about alpha in the trial direction (Hessian)
J = derivative(F, damage, damage_trial)


# loading and initialization of vectors to store time datas
load_multipliers = np.linspace(userpar["load_min"],userpar["load_max"],userpar["load_steps"])
energies = np.zeros((len(load_multipliers),4))
iterations = np.zeros((len(load_multipliers),2))
forces = np.zeros((len(load_multipliers), 2))

file_results = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
file_results.parameters["rewrite_function_mesh"] = False
file_results.parameters["functions_share_mesh"] = True
file_results.parameters["flush_output"] = True

# write the parameters to file
File(savedir+"/parameters.xml") << userpar

# -----------------------------------------------------------------------------
# Solving 
# -----------------------------------------------------------------------------
# Define the damage problem
(alpha_0, a_0, s_0, p_0) = damage.split(deepcopy=True)
if userpar["MITC"] == "project":
    problem_damage = fem.ProjectedNonlinearProblem(V_damage_P, F, damage, damage_p, bcs=bc_damage, J=J)
    damage_ = damage_p
else:
    problem_damage = fem.FullNonlinearProblem(V_damage_F, F, damage, bcs=bc_damage, J=J)
    damage_ = damage

# Initialize the damage snes solver and set the bounds
solver_damage = PETScSNESSolver()
solver_damage.set_from_options()
snes = solver_damage.snes()
#snes.setType("vinewtonssls")
as_vec = lambda field:  as_backend_type(field.vector()).vec() 
snes.setSolution(as_vec(damage_))
snes.setVariableBounds(as_vec(damage_lb),as_vec(damage_ub))

# Iterate on the time steps and solve
for (i_t, t) in enumerate(load_multipliers):
    u_UU.t = t * ut
    # u_R1.t = t*ut
    if MPI.rank(MPI.comm_world) == 0:
      print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))
    # Alternate Mininimization
    # Initialization
    iteration = 1
    err_alpha = 1.0
    # Iterations
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # solve elastic problem
        solver_u.solve()
        if MPI.rank(MPI.comm_world) == 0:
            print("\n@@@@@@ elastic sub-problem is solved @@@@@@\n")
        # solve damage problem
        solver_damage.solve(problem_damage,damage_.vector())
        if MPI.rank(MPI.comm_world) == 0:
            print("\n@@@@@@ damage  sub-problem is solved @@@@@@\n")
        problem_damage.counter = 0
        # check error
        (alpha_1, a_1, s_1, p_1) = damage.split(deepcopy=True)
        alpha_error = alpha_1.vector() - alpha_0.vector()
        err_alpha   = alpha_error.norm('linf')
        # monitor the results
        if MPI.rank(MPI.comm_world) == 0:
            print("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iterations
        alpha_0.assign(alpha_1)
        iteration = iteration+1
    # updating the lower bound to account for the irreversibility
    if userpar["MITC"] == "project":
        assigner_P.assign(damage_lb,[project(alpha_1,V_alpha),a_lb])
    else:
        assigner_F.assign(damage_lb,[project(alpha_1,V_alpha),a_lb,s_lb,p_lb])

    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Dump solution to file
    damage.split()[2].rename("Gradient_Damamge", "s")
    # file_s.write(damage.split()[1],t)
    damage.split()[0].rename("Damage", "alpha")
    u.rename("Displacement", "u")
    file_results.write(damage.split()[0],t)
    file_results.write(damage.split()[1],t)
    file_results.write(u, t)

    iterations[i_t] = np.array([t,iteration])
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    #ene_new = assemble(elastic_energy_new)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])

    # Calculate the reation force resultant
    #forces[i_t] = np.array([t, assemble(sigma_deCom(L4r,mL2s,pL2s,u,alpha)[0,1]*ds(1))])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [%s,%s]"%(elastic_energy_value, surface_energy_value))
        #print("-----------------------------------------")
        #print("\n@@@@@@@ Elastic Energies @@@@@@@: [{}, {}, {}]".format(elastic_energy_value, ene_new, elastic_energy_value-ene_new ))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir+'/energies.txt', energies)
        np.savetxt(savedir+'/iterations.txt', iterations)
        #np.savetxt(savedir+'/forces.txt', forces)
        #print("Results saved in ", savedir)
    #save_timings(savedir)

# Plot energy and stresses
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(energies[:,0], energies[:,1])
    p2, = plt.plot(energies[:,0], energies[:,2])
    p3, = plt.plot(energies[:,0], energies[:,3])
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"],loc = "best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir+'/energies.pdf', transparent=True)
    plt.close()

# Plot reaction forces
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(forces[:,0], forces[:,1], 'b-', linewidth = 2)
    # p2, = plt.plot(forces[:,0], forces[:,2], 'r-', linewidth = 2)
    plt.legend([p1], ["upper"], loc="best", frameon=False)
    plt.xlabel('Displacement [mm]')
    plt.ylabel('Forces [N]')
    plt.savefig(savedir + '/forces.pdf', transparent=True)
    plt.close()    