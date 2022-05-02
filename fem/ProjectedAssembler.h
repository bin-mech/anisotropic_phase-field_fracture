// Copyright (C) 2015 Jack S. Hale
//
// This file is part of fenics-shells.
//
// fenics-shells is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// fenics-shells is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with fenics-shells. If not, see <http://www.gnu.org/licenses/>.

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <dolfin/common/version.h>

#include <dolfin/fem/AssemblerBase.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/LocalAssembler.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>

namespace dolfin {

class ProjectedAssembler : public AssemblerBase
{
public:
    ProjectedAssembler () {}
    void assemble(GenericTensor& A,
                  GenericTensor& b,
                  const Form& a,
                  const Form& L,
                  const Form& a_dummy,
                  const Form& L_dummy,
                  const bool is_interpolation=false,
                  const bool a_is_symmetric=false);

    void reconstruct(Function& u_f_,
                     const Function& u_p_,
                     const Form& a,
                     const Form& L,
                     const bool is_interpolation=false,
                     const bool a_is_symmetric=false);

private:
    void check_arity(const Form& a,
                     const Form& L);
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        EigenRowMajorMatrix_t;
};

void ProjectedAssembler::check_arity(const Form& a,
                                     const Form& L) {
    // Check that a is a bilinear form
    if (a.rank() != 2)
    {
        dolfin_error("SystemAssembler.cpp",
                     "assemble system",
                     "expected a bilinear form for a");
    }

    // Check that L is a linear form
    if (L.rank() != 1)
    {
        dolfin_error("SystemAssembler.cpp",
                     "assemble system",
                     "expected a linear form for L");
    }
}

std::shared_ptr<const MeshFunction<std::size_t>>
_pick_one_meshfunction(std::string name,
                       std::shared_ptr<const MeshFunction<std::size_t>> a,
                       std::shared_ptr<const MeshFunction<std::size_t>> b)
{
    if ((a && b) && a != b)
    {
        warning("Bilinear and linear forms do not have same %s subdomains \
                 in ProjectedAssembler. Taking %s subdomains from bilinear form",
                 name.c_str(), name.c_str());
    }
    return a ? a: b;
}

void ProjectedAssembler::reconstruct(Function& u_f,
                                     const Function& u_p,
                                     const Form& a,
                                     const Form& L,
                                     const bool is_interpolation,
                                     const bool a_is_symmetric)
{
    AssemblerBase::check(a);
    AssemblerBase::check(L);

    check_arity(a, L);

    UFC a_ufc(a); UFC L_ufc(L);

    std::shared_ptr<const dolfin::Mesh> mesh = a.mesh();
    dolfin_assert(mesh);

    // Update off-process coefficients
    typedef std::vector<std::shared_ptr<const GenericFunction>> coefficient_t;
    const coefficient_t coefficients_a = a.coefficients();
    const coefficient_t coefficients_L = L.coefficients();

    std::shared_ptr<const MeshFunction<std::size_t>> cell_domains
        = _pick_one_meshfunction("cell_domains", a.cell_domains(),
                                 L.cell_domains());
    std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains
        = _pick_one_meshfunction("exterior_facet_domains",
                                 a.exterior_facet_domains(),
                                 L.exterior_facet_domains());
    std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains
        = _pick_one_meshfunction("interior_facet_domains",
                                 a.interior_facet_domains(),
                                 L.interior_facet_domains());

    std::size_t rows = a.function_space(0)->dofmap()->max_element_dofs();
    std::size_t cols = a.function_space(1)->dofmap()->max_element_dofs();

    dolfin_assert(rows == cols);

    std::size_t projected_rows = u_p.function_space()->dofmap()->max_element_dofs();

    std::shared_ptr<const FiniteElement> projected_fe
        = u_p.function_space()->element();
    std::shared_ptr<const FiniteElement> full_fe
        = a.function_space(0)->element();

    std::size_t full_sub_elements = full_fe->num_sub_elements();
    std::size_t projected_sub_elements = projected_fe->num_sub_elements();
    std::size_t num_projected_subspaces
        = full_sub_elements - projected_sub_elements;
    dolfin_assert(num_projected_subspaces == 2);

    std::vector<std::size_t> index_2 = {full_sub_elements - 2};
    std::shared_ptr<const FiniteElement> element_2 = full_fe->extract_sub_element(index_2);
    std::size_t size_2 = element_2->space_dimension();

    std::vector<std::size_t> index_3 = {full_sub_elements - 1};
    std::shared_ptr<const FiniteElement> element_3 = full_fe->extract_sub_element(index_3);
    std::size_t size_3 = element_3->space_dimension();

    // dofs 1
    std::size_t offset_1 = 0;
    std::size_t size_1 = projected_rows;
    // dofs 2
    std::size_t offset_2 = offset_1 + size_1;
    // dofs 3
    std::size_t offset_3 = offset_2 + size_2;

    const GenericDofMap* u_p_dofmap = u_p.function_space()->dofmap().get();
    const GenericDofMap* u_f_dofmap = u_f.function_space()->dofmap().get();

    EigenRowMajorMatrix_t u_f_cell(rows, 1);
    EigenRowMajorMatrix_t u_p_cell(projected_rows, 1);
    EigenRowMajorMatrix_t A_cell(rows, cols);
    EigenRowMajorMatrix_t b_cell(rows, 1);

    // Perform block splitting on bilinear form
    auto A_11 = A_cell.block(offset_1, offset_1, size_1, size_1);
    auto A_13 = A_cell.block(offset_1, offset_3, size_1, size_3);
    auto A_22 = A_cell.block(offset_2, offset_2, size_2, size_2);
    auto A_23 = A_cell.block(offset_2, offset_3, size_2, size_3);
    auto A_31 = A_cell.block(offset_3, offset_1, size_3, size_1);
    auto A_32 = A_cell.block(offset_3, offset_2, size_3, size_2);

    // and on linear form
    auto b_1 = b_cell.block(offset_1, 0, size_1, 1);
    auto b_2 = b_cell.block(offset_2, 0, size_2, 1);
    auto b_3 = b_cell.block(offset_3, 0, size_3, 1);

    std::vector<double> vertex_coordinates;
    ufc::cell ufc_cell;
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        dolfin_assert(!cell->is_ghost());
        cell->get_vertex_coordinates(vertex_coordinates);

        LocalAssembler::assemble(A_cell, a_ufc, vertex_coordinates, ufc_cell, *cell,
                                 cell_domains.get(),
                                 exterior_facet_domains.get(),
                                 interior_facet_domains.get());
        LocalAssembler::assemble(b_cell, L_ufc, vertex_coordinates, ufc_cell, *cell,
                                 cell_domains.get(),
                                 exterior_facet_domains.get(),
                                 interior_facet_domains.get());

        // Get primal solution on this cell
        auto u_p_dofs = u_p_dofmap->cell_dofs(cell->index());
        u_p.vector()->get_local(u_p_cell.data(), u_p_cell.rows(), u_p_dofs.data());

        // Reconstruct solution using block linear algebra
        // direct assignment for u_1
        u_f_cell.block(offset_1, 0, size_1, 1) = u_p_cell;
        if (is_interpolation) {
            // local interpolation for u_2
            // Do not fully understand negative signs here.
            u_f_cell.block(offset_2, 0, size_2, 1) =
                -(b_3 - A_31*u_p_cell);
            // local interpolation for u_3. noalias because rhs
            // does not intefere with assign on lhs.
            u_f_cell.block(offset_3, 0, size_3, 1).noalias() =
                -(b_2 - A_22*u_f_cell.block(offset_2, 0, size_2, 1));
        } else {
            // local solve for u_2
            u_f_cell.block(offset_2, 0, size_2, 1) =
                A_32.fullPivLu().solve(b_3 - A_31*u_p_cell);
            // local solve for u_3. noalias because rhs
            // does not intefere with assign on lhs.
            u_f_cell.block(offset_3, 0, size_3, 1).noalias() =
                A_23.fullPivLu().solve(b_2 - A_22*u_f_cell.block(offset_2, 0, size_2, 1));
        }

        auto u_f_dofs = u_f_dofmap->cell_dofs(cell->index());
        u_f.vector()->set_local(u_f_cell.data(), u_f_cell.rows(), u_f_dofs.data());
    }
    u_f.vector()->apply("insert");
}

void ProjectedAssembler::assemble(GenericTensor& A,
                                  GenericTensor& b,
                                  const Form& a, const Form& L,
                                  const Form& a_dummy,
                                  const Form& L_dummy,
                                  const bool is_interpolation,
                                  const bool a_is_symmetric)
{
    AssemblerBase::check(a);
    AssemblerBase::check(L);

    check_arity(a, L);

    UFC a_ufc(a); UFC L_ufc(L);

    std::shared_ptr<const dolfin::Mesh> mesh = a.mesh();
    dolfin_assert(mesh);

    // Update off-process coefficients
    typedef std::vector<std::shared_ptr<const GenericFunction>> coefficient_t;
    const coefficient_t coefficients_a = a.coefficients();
    const coefficient_t coefficients_L = L.coefficients();

    std::shared_ptr<const MeshFunction<std::size_t>> cell_domains
        = _pick_one_meshfunction("cell_domains", a.cell_domains(),
                                 L.cell_domains());
    std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains
        = _pick_one_meshfunction("exterior_facet_domains",
                                 a.exterior_facet_domains(),
                                 L.exterior_facet_domains());
    std::shared_ptr<const MeshFunction<std::size_t>> interior_facet_domains
        = _pick_one_meshfunction("interior_facet_domains",
                                 a.interior_facet_domains(),
                                 L.interior_facet_domains());

    // We assemble on the sparsity pattern of the dummy forms.
    std::vector<const GenericDofMap*> a_dofmaps;
    for (std::size_t i = 0; i < a_dummy.rank(); ++i) {
        a_dofmaps.push_back(a_dummy.function_space(i)->dofmap().get());
    }
    std::vector<ArrayView<const dolfin::la_index>> a_dofs(a_dummy.rank());

    std::vector<const GenericDofMap*> L_dofmaps;
    for (std::size_t i = 0; i < L_dummy.rank(); ++i) {
        L_dofmaps.push_back(a_dummy.function_space(i)->dofmap().get());
    }
    std::vector<ArrayView<const dolfin::la_index>> L_dofs(L_dummy.rank());

    std::size_t rows = a.function_space(0)->dofmap()->max_element_dofs();
    std::size_t cols = a.function_space(1)->dofmap()->max_element_dofs();

    dolfin_assert(rows == cols);

    std::size_t projected_rows
        = a_dummy.function_space(0)->dofmap()->max_element_dofs();
    std::size_t projected_cols = a_dummy.function_space(1)->dofmap()->max_element_dofs();

    dolfin_assert(projected_rows == projected_cols);

    std::shared_ptr<const FiniteElement> projected_fe
        = a_dummy.function_space(0)->element();
    std::shared_ptr<const FiniteElement> full_fe
        = a.function_space(0)->element();

    std::size_t full_sub_elements = full_fe->num_sub_elements();
    std::size_t projected_sub_elements = projected_fe->num_sub_elements();
    std::size_t num_projected_subspaces
        = full_sub_elements - projected_sub_elements;
    dolfin_assert(num_projected_subspaces == 2);

    std::vector<std::size_t> index_2 = {full_sub_elements - 2};
    std::shared_ptr<const FiniteElement> element_2 = full_fe->extract_sub_element(index_2);
    std::size_t size_2 = element_2->space_dimension();

    std::vector<std::size_t> index_3 = {full_sub_elements - 1};
    std::shared_ptr<const FiniteElement> element_3 = full_fe->extract_sub_element(index_3);
    std::size_t size_3 = element_3->space_dimension();

    // dofs 1
    std::size_t offset_1 = 0;
    std::size_t size_1 = projected_rows;
    // dofs 2
    std::size_t offset_2 = offset_1 + size_1;
    // dofs 3
    std::size_t offset_3 = offset_2 + size_2;

    // We initialise the tensors to the sparsity pattern of the dummy forms.
    init_global_tensor(A, a_dummy);
    init_global_tensor(b, L_dummy);

    // For LocalAssembler
    EigenRowMajorMatrix_t A_cell(rows, cols);
    EigenRowMajorMatrix_t b_cell(rows, 1);

    // The below are views into the underlying cell operators, so
    // we can construct them before the loop.
    // Perform block splitting on bilinear form.
    auto A_11 = A_cell.block(offset_1, offset_1, size_1, size_1);
    auto A_13 = A_cell.block(offset_1, offset_3, size_1, size_3);
    auto A_22 = A_cell.block(offset_2, offset_2, size_2, size_2);
    auto A_23 = A_cell.block(offset_2, offset_3, size_2, size_3);
    auto A_31 = A_cell.block(offset_3, offset_1, size_3, size_1);
    auto A_32 = A_cell.block(offset_3, offset_2, size_3, size_2);

    // and on linear form
    auto b_1 = b_cell.block(offset_1, 0, size_1, 1);
    auto b_2 = b_cell.block(offset_2, 0, size_2, 1);
    auto b_3 = b_cell.block(offset_3, 0, size_3, 1);

    // Preallocations
    EigenRowMajorMatrix_t A_cell_projected(rows, cols);
    EigenRowMajorMatrix_t b_cell_projected(rows, 1);

    std::vector<double> vertex_coordinates;
    ufc::cell ufc_cell;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        dolfin_assert(!cell->is_ghost());
        cell->get_vertex_coordinates(vertex_coordinates);

        LocalAssembler::assemble(A_cell, a_ufc, vertex_coordinates, ufc_cell, *cell,
                                 cell_domains.get(),
                                 exterior_facet_domains.get(),
                                 interior_facet_domains.get());
        LocalAssembler::assemble(b_cell, L_ufc, vertex_coordinates, ufc_cell, *cell,
                                 cell_domains.get(),
                                 exterior_facet_domains.get(),
                                 interior_facet_domains.get());

        // Compute projections
        // TODO: Optimise.
        if (is_interpolation) {
            A_cell_projected = A_11
                + A_13*A_22*A_31;
            b_cell_projected = b_1
                + A_13*A_22*b_3
                - A_13*b_2;
        } else {
            A_cell_projected = A_11
                + A_13*A_23.inverse()*A_22*A_32.inverse()*A_31;
            b_cell_projected = b_1
                + A_13*A_23.inverse()*A_22*A_32.inverse()*b_3
                - A_13*A_23.inverse()*b_2;
            // std::cout << A_32.inverse().array();

        }

        for (std::size_t i = 0; i < 2; ++i)
        {
            auto dofmap = a_dofmaps[i]->cell_dofs(cell->index());
            a_dofs[i].set(dofmap.size(), dofmap.data());
        }
        for (std::size_t i = 0; i < 1; ++i)
        {
            auto dofmap = a_dofmaps[i]->cell_dofs(cell->index());
            L_dofs[i].set(dofmap.size(), dofmap.data());
        }

        A.add_local(A_cell_projected.data(), a_dofs);
        b.add_local(b_cell_projected.data(), L_dofs);

    }

    if (finalize_tensor) {
        A.apply("add");
        b.apply("add");
    }
}

namespace py = pybind11;

PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<ProjectedAssembler, std::shared_ptr<ProjectedAssembler>, AssemblerBase>
        (m, "ProjectedAssembler", "Class ProjectedAssembler")
        .def(py::init<>())
        .def("assemble", (void (ProjectedAssembler::*)(dolfin::GenericTensor&,
                                                       dolfin::GenericTensor&,
                                                       const dolfin::Form&,
                                                       const dolfin::Form&,
                                                       const dolfin::Form&,
                                                       const dolfin::Form&,
                                                       const bool,
                                                       const bool)) &ProjectedAssembler::assemble)
        .def("reconstruct", (void (ProjectedAssembler::*)(dolfin::Function&,
                                                          const dolfin::Function&,
                                                          const dolfin::Form&,
                                                          const dolfin::Form&,
                                                          const bool,
                                                          const bool)) &ProjectedAssembler::reconstruct);
}
}