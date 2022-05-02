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
import dolfin.cpp as cpp
import ufl

class FullFunctionSpace(dolfin.FunctionSpace):
    def __init__(self, *args, **kwargs):
        dolfin.FunctionSpace.__init__(self, *args, **kwargs)

class ProjectedFunctionSpace(dolfin.FunctionSpace):
    def __init__(self, *args, **kwargs):
        self._num_projected_subspaces = kwargs.pop('num_projected_subspaces')
        if self._num_projected_subspaces < 1:
            cpp.dolfin_error("functionspace.py",
                             "create projection function space",
                             "Illegal keyword argument, num_projected_subspaces >= 1")

        if not isinstance(args[1], ufl.MixedElement):
            cpp.dolfin_error("functionspace.py",
                             "create projection function space",
                             "Illegal argument, not a MixedElement" + \
                             str(args[1]))

        self._full_element = args[1]
        self._projected_element = ufl.MixedElement(self._full_element.sub_elements()[0:-self._num_projected_subspaces])

        # Construct projected space (self)
        dolfin.FunctionSpace.__init__(self, args[0], self._projected_element, **kwargs)
        # Construct projected space
        self._full_function_space = FullFunctionSpace(self.mesh(), self._full_element, **kwargs)
        self._full_function_space._projected_function_space = self

    @property
    def projected_space(self):
        return self

    @property
    def P(self):
        return self.projected_space

    @property
    def full_space(self):
        return self._full_function_space

    @property
    def F(self):
        return self.full_space

