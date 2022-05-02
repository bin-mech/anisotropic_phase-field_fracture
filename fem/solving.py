#
# This file is part of fenics-shells.
#
# fenics-shells is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fenics-shells is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fenics-shells. If not, see <http://www.gnu.org/licenses/>.

import dolfin as df
from dolfin.fem.assembling import _create_dolfin_form

import os
pwd = os.path.dirname(__file__)
with open(pwd + "/ProjectedAssembler.h", "r") as f:
    projected_assembler_code = f.read()
    cpp = df.compile_cpp_code(projected_assembler_code)

from .assembling import assemble, projected_assemble
from .functionspace import ProjectedFunctionSpace, FullFunctionSpace
def project(u, V):
    if isinstance(V, FullFunctionSpace):
        u_V_F = Function(V)
        _reconstruct_full_space(u, u_V_F)
        return u_V_F
    elif isinstance(V, ProjectedFunctionSpace):
        raise NotImplementedError("You passed a ProjectedFunctionSpace instead of a FullFunctionSpace.")
    elif isinstance(V, df.FunctionSpace):
        u_V = df.project(u, V)
        return u_V
    else:
        raise TypeError("V should be a ProjectedFunctionSpace or a FunctionSpace.")


def reconstruct_full_space(u_f, u_p, a, L, is_interpolation=False, a_is_symmetric=False, form_compiler_parameters=None):
    """
    Given a Function on a projected space :math:`u_p \in U_P` and a function in
    the full space :math:`u_f \in U_F: such that :math:`U_P \subset U_F`,
    reconstruct the variable u_f on the full space via direct copy of the
    matching subfunctions shared between U_F and U_P and the local solution
    of the original problem a == L.

    Args:
        u_f: DOLFIN Function on FullFunctionSpace.
        u_p: DOLFIN Function on ProjectedFunctionSpace.
        a: Bilinear form.
        L: Bilinear form.

    Returns:
        u_f: DOLFIN Function on FullFunctionSpace.
    """
    U_P = u_p.function_space()
    U_F = u_f.function_space()

    assert(U_P.num_sub_spaces() < U_F.num_sub_spaces())

    a_dolfin = _create_dolfin_form(a, form_compiler_parameters)
    L_dolfin = _create_dolfin_form(L, form_compiler_parameters)

    assembler = cpp.ProjectedAssembler()
    assembler.reconstruct(u_f.cpp_object(), u_p.cpp_object(), a_dolfin, L_dolfin, is_interpolation, a_is_symmetric)

    return u_f

class ProjectedNonlinearProblem(df.NonlinearProblem):
    def __init__(self, U_P, F, u_f_, u_p_, bcs=None, J=None):
        """
        Allows the solution of non-linear problems using the projected_assemble
        routines. Can be used with any of the built-in non-linear solvers
        in DOLFIN.

        Args:

        """
        df.NonlinearProblem.__init__(self)

        self.U_P = U_P
        self.U_F = U_P.full_space
        self.F_form = F
        self.J_form = J
        self.u_f_ = u_f_
        self.u_p_ = u_p_
        self.bcs = bcs

        self.x_p_prev = u_p_.vector().copy()
        self.x_p_prev.zero()
        self.dx_f = df.Function(self.U_F)

    def form(self, A, P, b, x_p):
        # To do the reconstruction of the increment on the full space, we need
        # the increment on the projected space. This is not accessible through
        # the current DOLFIN interfaces. So, we have to recalculate the
        # projected increment, reconstruct the full increment, then perform
        # the Newton step on the full space.

        # Recalculate dx_p. Was available inside NonlinearSolver, but not
        # accessible here.
        dx_p_vector = self.x_p_prev - x_p
        dx_p = df.Function(self.U_P, dx_p_vector)

        # Reconstruct according to J == F
        reconstruct_full_space(self.dx_f, dx_p, self.J_form, self.F_form)

        # Newton update
        #if isinstance(alpha_p,Function):
        self.u_f_.vector()[:] -= self.dx_f.vector()

        # Store x_p for use in next Newton iteration
        self.x_p_prev[:] = x_p

        # Then as normal...
        assemble(self.U_P, self.J_form, self.F_form, A=A, b=b,
                       is_interpolation=False,
                       a_is_symmetric=False,
                       form_compiler_parameters=None,
                       add_values=False,
                       finalize_tensor=True,
                       keep_diagonal=False,
                       backend=None)

        for bc in self.bcs:
            bc.apply(A, b, x_p)

    def F(self, b, x_p):
        pass

    def J(self, A, x_p):
        pass

class FullNonlinearProblem(df.NonlinearProblem):
    def __init__(self, U_F, F, u_f_, bcs=None, J=None):
        """
        Allows the solution of non-linear problems using the projected_assemble
        routines. Can be used with any of the built-in non-linear solvers
        in DOLFIN.

        Args:

        """
        df.NonlinearProblem.__init__(self)
        self.U_F = U_F
        self.F_form = F
        self.J_form = J
        self.u_f_ = u_f_
        self.bcs = bcs

    def form(self, A, P, b, x_p):
        df.assemble_system(self.J_form,self.F_form,self.bcs,A_tensor=A,b_tensor=b)

    def F(self, b, x_p):
        pass

    def J(self, A, x_p):
        pass