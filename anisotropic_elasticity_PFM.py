#  
# =============================================================================
# FEniCS code  Variational Fracture Mechanics
# =============================================================================
# 
# A static solution of the variational fracture mechanics problems  
# using the regularization two-fold anisotropic damage model
#
# author: Bin Li (bin.l@gtiit.edu.cn), 
#
# date: 07/16/2021
# ----------------
# References:
# Ref1: He, Q-C., and Q. Shao. "Closed-form coordinate-free decompositions of the 
#       two-dimensional strain and stress for modeling tension–compression 
#       dissymmetry." Journal of Applied Mechanics 86.3 (2019): 031007.


# ----------------------------------------------------------------------------
from __future__ import division
from dolfin import *
from mshr import *
from ufl import ne, transpose

import argparse
import math
import os
import shutil
import sympy
import sys
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Parameters for DOLFIN and SOLVER 
# ----------------------------------------------------------------------------
set_log_level(LogLevel.ERROR)  # log level
# set some dolfin specific parameters
info(parameters,True)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

# -----------------------------------------------------------------------------
# parameters of the solvers
"""
solver_u_parameters = {"nonlinear_solver": "newton",
                       "newton_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 100,
                                          "absolute_tolerance": 1e-8,
                                          "relative_tolerance": 1e-8,
                                          "report": True,
                                          "error_on_nonconvergence": True}}    
"""                                              
solver_u_parameters  = {"nonlinear_solver": "snes",
                        "symmetric": True,
                        "snes_solver": {"linear_solver": "mumps", #"mumps",
                                        "method": "newtonls", #"newtontr", "newtonls"
                                        "line_search":"basic", # "basic", "bt", "l2", "cp", "nleqerr"
                                        "preconditioner": "hypre_amg", 
                                        "maximum_iterations": 200,
                                        "absolute_tolerance": 1e-10,
                                        "relative_tolerance": 1e-10,
                                        "solution_tolerance": 1e-10,
                                        "report": True,
                                        "error_on_nonconvergence": True}}   

# parameters of the PETSc/Tao solver used for the alpha-problem
tao_solver_parameters = {"maximum_iterations": 200,
                         "report": False,
                         "line_search": "more-thuente",
                         "linear_solver": "mumps",
                         "method": "tron",
                         "gradient_absolute_tol": 1e-8,
                         "gradient_relative_tol": 1e-8,
                         "error_on_nonconvergence": True}

# -----------------------------------------------------------------------------
# set the user parameters
parameters.parse()
userpar = Parameters("user")
userpar.add("load_min",0.0e-2)
userpar.add("load_max",2.5e-2)
userpar.add("load_steps",251)
userpar.add("theta",0)

userpar.parse()

# ----------------------------------------------------------------------------
# Parameters for surface energy and materials
# ----------------------------------------------------------------------------
E  = 1.2e4
nu = 0.3

mu = E/(2.0*(1.0+nu))
lmbda = E*nu/(1.0-2.0*nu)/(1.0+nu)  # plane strain

# Material constant
# stiffness matrix in Voigt notation
L1111 = 2.*mu+lmbda 
L1122 = lmbda 
L1112 = 0.0
L2222 = 2.*mu+lmbda 
L2212 = 0.0
L1212 = mu 

L4 = np.matrix([[L1111, L1122, L1112],\
                [L1122, L2222, L2212],\
                [L1112, L2212, L1212]]) 

# -----------------------------------------------------------------------------
# 3D second-order stiffness matrix (see Ref1)
L2 = np.matrix([[L1111, L1122, sqrt(2.0)*L1112],\
                [L1122, L2222, sqrt(2.0)*L2212],\
                [sqrt(2.0)*L1112, sqrt(2.0)*L2212, 2.0*L1212]]) 


# -----------------------------------------------------------------------------
# rotation matrix for stiffness matrix in Voigt notation
theta0 = userpar["theta"]*np.pi/180
K = np.matrix([[np.cos(theta0)**2, np.sin(theta0)**2 ,  2.0*np.cos(theta0)*np.sin(theta0)], \
               [np.sin(theta0)**2, np.cos(theta0)**2 , -2.0*np.cos(theta0)*np.sin(theta0)], \
               [-np.cos(theta0)*np.sin(theta0), np.cos(theta0)*np.sin(theta0) , np.cos(theta0)**2-np.sin(theta0)**2]])
L4matr = np.matmul(np.matmul(K,L4), np.transpose(K))

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
    Cr23 = Cmatr[1,2]
    Cr33 = Cmatr[2,2]
    return as_matrix([[Cr11, Cr12, Cr13], [Cr12, Cr11, Cr23], [Cr13, Cr23, Cr33]])

L4r = npmat2matrix(L4matr)

Gc = Constant(1.4e-3)
k_ell = Constant(1.e-6)  # residual stiffness

# Loading Parameters
ut      = 1   # reference value for the loading (imposed displacement)

# Numerical parameters of the alternate minimization
maxiteration = 2000
AM_tolerance = 1e-4
# Geometry paramaters
L         = 15
H         = 30
H1        = H/2
H2        = -H/2

ell       = Constant(0.12) # damage paramaters
modelname = "hole_sample"
meshname  = modelname+"-mesh.xdmf"
simulation_params = "decomposition_%.1f_L_%.4f_H_%.4f_ell_%.4f" %(userpar["theta"], L, H, ell)
savedir   = "output/"+modelname+"/"+simulation_params+"/"

if MPI.rank(MPI.comm_world) == 0:
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)

mesh = Mesh("../meshes/hole_sample.xml")
geo_mesh  = XDMFFile(MPI.comm_world, savedir+meshname)
geo_mesh.write(mesh)


mesh.init()
ndim = mesh.geometry().dim()  # get number of space dimensions
if MPI.rank(MPI.comm_world) == 0:
    print ("the dimension of mesh: {0:2d}".format(ndim))

# ----------------------------------------------------------------------------
# Strain and stress and Constitutive functions of the damage model
# ----------------------------------------------------------------------------
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
    return voigt2stress(dot(L4r, strain2voigt(eps(v)))) # Plane Strain

# Constitutive functions of the damage model
def w(alpha):
    return alpha

def a(alpha):
    return (1.0-alpha)**2

# ----------------------------------------------------------------------------
# Variational formulation 
# ----------------------------------------------------------------------------
# Create function space for 2D elasticity + Damage
V_u     = VectorFunctionSpace(mesh, "Lagrange", 1)
V_alpha = FunctionSpace(mesh, "Lagrange", 1)
Stress  = TensorFunctionSpace(mesh,'DG',0)
# Define the function, test and trial fields
u       = Function(V_u, name="Displacement")
du      = TrialFunction(V_u)
v       = TestFunction(V_u)
alpha   = Function(V_alpha, name="Damage")
dalpha  = TrialFunction(V_alpha)
beta    = TestFunction(V_alpha)

# --------------------------------------------------------------------
# Dirichlet boundary condition
# Impose the displacements field given by asymptotic expansion of crack tip
# --------------------------------------------------------------------
upper = CompiledSubDomain("near(x[1], %s)"%H1)
lower = CompiledSubDomain("near(x[1], %s)"%H2)

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
upper.mark(boundaries, 1)
lower.mark(boundaries, 2)
# left_upper: ds(1), right: ds(2)
ds = Measure("ds",subdomain_data=boundaries) 

# Displacement
u_R1 = Expression(("0","t"), t = 0.0, degree=0)
bcu_1 = DirichletBC(V_u, u_R1, boundaries, 1)
bcu_2 = DirichletBC(V_u, ("0", "0"), boundaries, 2)
bc_u = [bcu_1, bcu_2]

# Damage
bcalpha_1 = DirichletBC(V_alpha, 0.0, boundaries, 1)
bcalpha_2 = DirichletBC(V_alpha, 0.0, boundaries, 2)
bc_alpha = [bcalpha_1]


# --------------------------------------------------------------------
# Define the energy functional of elastic problem
# --------------------------------------------------------------------
u_0 = interpolate(Expression(("0.","0."), degree=0), V_u)  # previous (known) displacements

# eigenvalues of 3D second-order stiffness matrix 
lmbda1 = (L2[0,0]+L2[0,1])
lmbda2 = (L2[0,0]-L2[0,1])
lmbda3 = L2[2,2]

# eigenprojectors for L2
A10 = np.matrix([[0.5, 0.5, 0.0], \
                 [0.5, 0.5, 0.0], \
                 [0.0, 0.0, 0.0]])

A20 = np.matrix([[ 0.5,-0.5, 0.0], \
                 [-0.5, 0.5, 0.0], \
                 [ 0.0, 0.0, 0.0]])
 
A30 = np.matrix([[0.0, 0.0, 0.0], \
                 [0.0, 0.0, 0.0], \
                 [0.0, 0.0, 1.0]])

A11 = np.matmul(np.matmul(R,A10), np.transpose(R))

A21 = np.matmul(np.matmul(R,A20), np.transpose(R))

A31 = np.matmul(np.matmul(R,A30), np.transpose(R))

pL2s0 = sqrt(lmbda1)*A11 + sqrt(lmbda2)*A21 + sqrt(lmbda3)*A31 

mL2s0 = 1.0/sqrt(lmbda1)*A11 + 1.0/sqrt(lmbda2)*A21 + 1.0/sqrt(lmbda3)*A31 

pL2s = npmat2matrix(pL2s0)
mL2s = npmat2matrix(mL2s0)

L2r = np.matrix([[L4matr[0,0], L4matr[0,1], sqrt(2.0)*L4matr[0,2]],\
                 [L4matr[1,0], L4matr[1,1], sqrt(2.0)*L4matr[1,2]],\
                 [sqrt(2.0)*L4matr[2,0], sqrt(2.0)*L4matr[2,1], 2.0*L4matr[2,2]]]) 

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
    return (a(alpha)+k_ell)*sigma_deCom_plus(L4r,mL2s,pL2s,u)+sigma_deCom_minus(L4r,mL2s,pL2s,u)

elastic_energy = ((a(alpha)+k_ell)*energy_active(L4r,mL2s,pL2s,u)+energy_passive(L4r,mL2s,pL2s,u))*dx


def sigma(u, alpha):
    return (a(alpha)+k_ell)*sigma_0(u)

elastic_energy_new = 0.5*inner(sigma(u, alpha), eps(u))*dx
body_force        = Constant((0., 0.))
external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy-external_work

# Weak form of elasticity problem
E_u  = derivative(elastic_potential, u, v)
# Writing tangent problems in term of test and trial functions for matrix assembly
E_du = derivative(E_u, u, du)

# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(E_u, u, bc_u, J=E_du)
# Set up the solvers                                        
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)
# info(solver_u.parameters, True)

# --------------------------------------------------------------------
# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Fenics forms for the energies

alpha_0 = interpolate(Expression("0.", degree=0), V_alpha)  # initial (known) alpha
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell+ell*inner(grad(alpha), grad(alpha)))*dx
damage_functional = elastic_potential + dissipated_energy

# Compute directional derivative about alpha in the direction of beta (Gradient)
E_alpha       = derivative(damage_functional, alpha, beta)
# Compute directional derivative about alpha in the direction of dalpha (Hessian)
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# --------------------------------------------------------------------
# Implement the box constraints for damage field
# --------------------------------------------------------------------
# Variational problem for the damage (non-linear to use variational inequality solvers of petsc)
# Define the minimisation problem by using OptimisationProblem class
class DamageProblem(OptimisationProblem):

    def __init__(self,f,gradf,alpha,J,bcs):
        OptimisationProblem.__init__(self)
        self.total_energy = f
        self.Dalpha_total_energy = gradf
        self.J_alpha = J
        self.alpha = alpha
        self.bc_alpha = bcs

    def f(self, x):
        self.alpha.vector()[:] = x
        return assemble(self.total_energy)

    def F(self, b, x):
        self.alpha.vector()[:] = x
        assemble(self.Dalpha_total_energy, b)
        for bc in self.bc_alpha:
            bc.apply(b)

    def J(self, A, x):
        self.alpha.vector()[:] = x
        assemble(self.J_alpha, A)
        for bc in self.bc_alpha:
            bc.apply(A)

damage_problem = DamageProblem(damage_functional,E_alpha,alpha,E_alpha_alpha,bc_alpha)

# Set up the solvers                                        
solver_alpha  = PETScTAOSolver()
solver_alpha.parameters.update(tao_solver_parameters)
#alpha_lb = interpolate(Expression("x[0]<=0.5 & near(x[1], 0.0, 0.1 * hsize) ? 1.0 : 0.0", \
#                       hsize = hsize, degree=0), V_alpha)
alpha_lb = interpolate(Expression("0.", degree=0), V_alpha)  # lower bound, set to 0
alpha_ub = interpolate(Expression("1.", degree=0), V_alpha)  # upper bound, set to 1

# loading and initialization of vectors to store time datas
load_multipliers  = np.linspace(userpar["load_min"], userpar["load_max"], userpar["load_steps"])
energies          = np.zeros((len(load_multipliers), 4))
iterations        = np.zeros((len(load_multipliers), 2))
forces            = np.zeros((len(load_multipliers), 2))

# set the saved data file name
file_results = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
file_results.parameters["rewrite_function_mesh"] = False
file_results.parameters["functions_share_mesh"] = True
file_results.parameters["flush_output"] = True

"""
file_alpha  = XDMFFile(MPI.comm_world, savedir + "/alpha.xdmf")
file_alpha.parameters["rewrite_function_mesh"]      = False
file_alpha.parameters["flush_output"]               = True
"""

# write the parameters to file
File(savedir+"/parameters.xml") << userpar

# ----------------------------------------------------------------------------
# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    u_R1.t = -t * ut
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
        # solve damage problem with box constraint 
        solver_alpha.solve(damage_problem, alpha.vector(), alpha_lb.vector(), alpha_ub.vector())
        # test error
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')
        # monitor the results
        if MPI.rank(MPI.comm_world) == 0:
          print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iteration
        u_0.assign(u)
        alpha_0.assign(alpha)
        iteration = iteration + 1
    # updating the lower bound to account for the irreversibility
    alpha_lb.vector()[:] = alpha.vector()
    #STensor = project(sigma(u, alpha), Stress)
    alpha.rename("Damage", "alpha")
    u.rename("Displacement", "u")
    #STensor.rename("Stress", "sigma")
    # Dump solution to file 
    file_results.write(alpha, t)
    file_results.write(u, t)
    #file_results.write(STensor, t)
    # ----------------------------------------
    # Some post-processing
    # ----------------------------------------
    # Save number of iterations for the time step    
    iterations[i_t] = np.array([t, iteration])

    # Calculate the energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    #ene_new = assemble(elastic_energy_new)
    energies[i_t] = np.array([t, elastic_energy_value, surface_energy_value, elastic_energy_value+surface_energy_value])

    # Calculate the reation force resultant
    forces[i_t] = np.array([t, -assemble(sigma_deCom(L4r,mL2s,pL2s,u,alpha)[1 ,1]*ds(1))])

    if MPI.rank(MPI.comm_world) == 0:
        print("\nEnd of timestep {0:3d} with load multiplier {1:4f}".format(i_t, t))
        print("\nElastic and Surface Energies: [{0:6f},{1:6f}]".format(elastic_energy_value, surface_energy_value))
        print("\nElastic and Surface Energies: [{},{}]".format(elastic_energy_value, surface_energy_value))
        #print("-----------------------------------------")
        #print("\n@@@@@@@ Elastic Energies @@@@@@@: [{}, {}, {}]".format(elastic_energy_value, ene_new, elastic_energy_value-ene_new ))
        print("-----------------------------------------")
        # Save some global quantities as a function of the time
        np.savetxt(savedir + '/energies.txt', energies)
        np.savetxt(savedir + '/iterations.txt', iterations)
        np.savetxt(savedir + '/force.txt', forces)
# ----------------------------------------------------------------------------

# Plot energy and stresses
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(energies[:, 0], energies[:, 1])
    p2, = plt.plot(energies[:, 0], energies[:, 2])
    p3, = plt.plot(energies[:, 0], energies[:, 3])
    plt.legend([p1, p2, p3], ["Elastic", "Dissipated", "Total"], loc="best", frameon=False)
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    plt.savefig(savedir + '/energies.pdf', transparent=True)
    plt.close()


# Plot reaction forces
if MPI.rank(MPI.comm_world) == 0:
    p1, = plt.plot(forces[:,0], forces[:,1], 'b-', linewidth = 2)
    # p2, = plt.plot(forces[:,0], forces[:,2], 'r-', linewidth = 2)
    plt.legend([p1], ["upper"], loc="best", frameon=False)
    plt.xlabel('Displacement [mm]')
    plt.ylabel('Forces [kN]')
    plt.savefig(savedir + '/forces.pdf', transparent=True)
    plt.close()