# -*- coding: utf-8 -*-

# Copyright (C) 2015 Jack S. Hale
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

import dolfin
import dolfin.cpp
from dolfin import inner, dx, TrialFunction, TestFunction
from dolfin.fem.assembling import _create_dolfin_form, _create_tensor
from .functionspace import ProjectedFunctionSpace, FullFunctionSpace
import os
pwd = os.path.dirname(__file__)
with open(pwd + "/ProjectedAssembler.h", "r") as f:
    projected_assembler_code = f.read()
    cpp = dolfin.compile_cpp_code(projected_assembler_code)

def assemble(*args, **kwargs):
    """Pass-through for dolfin.assemble and fenics_shells.projected_assemble.

    If the first argument is an instance of ProjectedFunctionSpace it will call
    fenics_shells.projected_assemble, otherwise it will pass through to
    dolfin.assemble.
    """
    if isinstance(args[0], ProjectedFunctionSpace):
        A = projected_assemble(*args, **kwargs)
        return A
    else:
        A = dolfin.assemble(*args, **kwargs)
        return A

def projected_assemble(U_P, a, L, bcs=None, A=None, b=None,
                       is_interpolation=False,
                       a_is_symmetric=False,
                       form_compiler_parameters=None,
                       add_values=False,
                       finalize_tensor=True,
                       keep_diagonal=False,
                       backend=None):
    U_F = U_P.full_space

    if a is None or L is None:
        dolfin.cpp.dolfin_error("assembling.py", "assemble projected form",
                "Must pass bilinear form a and bilinear form L")

    if not isinstance(U_P, ProjectedFunctionSpace):
        dolfin.cpp.dolfin_error("assembling.py", "assemble projected form",
                "Expected U to be a ProjectedFunctionSpace")

    # We want to assemble our final tensor on P(U). So we make a dummy form for
    # initializing our sparse tensors. These forms are never assembled.
    u = TrialFunction(U_P)
    v = TestFunction(U_P)
    a_dummy_form = inner(u, v)*dx

    u = TrialFunction(U_P)
    v = TestFunction(U_P)
    f = dolfin.Constant([0.0]*v.ufl_shape[0])
    L_dummy_form = inner(f, v)*dx

    a_dummy_form_dolfin = _create_dolfin_form(a_dummy_form, form_compiler_parameters)
    L_dummy_form_dolfin = _create_dolfin_form(L_dummy_form, form_compiler_parameters)

    mpi_comm = a_dummy_form_dolfin.mesh().mpi_comm()
    A = _create_tensor(mpi_comm, a_dummy_form_dolfin, a_dummy_form_dolfin.rank(), backend, A)
    b = _create_tensor(mpi_comm, L_dummy_form_dolfin, L_dummy_form_dolfin.rank(), backend, b)

    # Now the forms that we use to assemble the cell tensors before static condensation.
    a_dolfin = _create_dolfin_form(a, form_compiler_parameters)
    L_dolfin = _create_dolfin_form(L, form_compiler_parameters)

    assembler = cpp.ProjectedAssembler()
    assembler.add_values = add_values
    assembler.finalize_tensor = finalize_tensor
    assembler.keep_diagonal = keep_diagonal
    assembler.assemble(A, b, a_dolfin, L_dolfin,
                       a_dummy_form_dolfin, L_dummy_form_dolfin,
                       is_interpolation, a_is_symmetric)

    return (A, b)
