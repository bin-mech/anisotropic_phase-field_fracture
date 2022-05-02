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

import os
import dolfin

# This gets the directory name of this file (__init__.py)
# so we can read in the cpp code
pwd = os.path.dirname(__file__)

# JIT the Projected Assembler
with open(pwd + "/ProjectedAssembler.h", "r") as f:
    projected_assembler_code = f.read()
cpp = dolfin.compile_cpp_code(projected_assembler_code)

from .assembling import projected_assemble, assemble
from .solving import project, ProjectedNonlinearProblem, reconstruct_full_space, FullNonlinearProblem
from .functionspace import FullFunctionSpace, ProjectedFunctionSpace
__all__ = ["cpp","FullFunctionSpace","ProjectedFunctionSpace","projected_assemble",
            "project", "ProjectedNonlinearProblem", "FullNonlinearProblem", "assemble", "reconstruct_full_space"]

