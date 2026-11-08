// Copyright (c) 2026 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#include <algorithm>
#include <gtest/gtest.h>
#include "../DenseTestModel.hpp"
#include "ingredients/hessian_models/quasi_newton/direct/LSR1Hessian.hpp"
#include "ingredients/inertia_correction_strategies/NoInertiaCorrection.hpp"
#include "ingredients/subproblem/Subproblem.hpp"
#include "ingredients/subproblem_solvers/COOLinearSystem.hpp"
#include "ingredients/subproblem_solvers/DirectSymmetricIndefiniteLinearSolver.hpp"
#include "ingredients/subproblem_solvers/LowRankCorrection.hpp"
#include "optimization/OptimizationProblem.hpp"
#include "options/DefaultOptions.hpp"
#include "options/Options.hpp"

using namespace uno;

namespace {
   class WorkspaceHessian: public LSR1Hessian {
   public:
      using LSR1Hessian::LSR1Hessian;
      void set_columns(size_t rank, double scale) {
         this->number_entries_in_memory = rank;
         this->U.fill(0.);
         if (0 < rank) {
            this->U.entry(0, 0) = scale;
            this->U.entry(1, 0) = scale / 2.;
            this->correction_scaling[0] = 1.;
         }
         if (1 < rank) {
            this->U.entry(0, 1) = scale / 4.;
            this->U.entry(1, 1) = scale;
            this->correction_scaling[1] = 1.;
         }
      }
   };

   class ScaledIdentitySolver: public DirectSymmetricIndefiniteLinearSolver<double> {
   public:
      explicit ScaledIdentitySolver(size_t dimension) { this->system.dimension = dimension; }
      double diagonal{1.};
      void initialize_memory() override { }
      void do_symbolic_analysis() override { }
      void do_numerical_factorization(bool) override { }
      void solve_indefinite_system(double* solution) override { this->solve_indefinite_system(this->system.rhs.data(), solution, 1); }
      void solve_indefinite_system(const double* rhs, double* solution, size_t nrhs) override {
         for (size_t i = 0; i < nrhs * this->system.dimension; ++i) solution[i] = rhs[i] / this->diagonal;
      }
      Inertia get_inertia() const override { return {this->system.dimension, 0, 0}; }
      size_t number_negative_eigenvalues() const override { return 0; }
      bool matrix_is_singular() const override { return false; }
      size_t rank() const override { return this->system.dimension; }
      LinearSystem& get_linear_system() override { return this->system; }
   private:
      COOLinearSystem system{0};
   };

   DenseTestModel make_model(size_t m) {
      return DenseTestModel("workspace", 2, m, ProblemType::NONLINEAR,
         [](const Vector<double>& x) { return (x[0] * x[0] + x[1] * x[1]) / 2.; },
         [](const Vector<double>& x, Vector<double>& g) { g = x; },
         [](const Vector<double>&, Vector<double>& c) { c.fill(0.); },
         [m](const Vector<double>&, double* j) { if (m) std::fill_n(j, 2 * m, 0.); },
         [](const Vector<double>&, double w, const Vector<double>&, double* h) { h[0] = w; h[1] = 0.; h[2] = w; },
         {-Inf, -Inf}, {Inf, Inf}, std::vector<double>(m), std::vector<double>(m), {1., 2.});
   }
} // namespace

TEST(WoodburyWorkspace, RepeatedUpdatesHandleChangedColumnsFactorsRankAndPadding) {
   Options options;
   DefaultOptions::load(options);
   const DenseTestModel model = make_model(0);
   WorkspaceHessian hessian(model, 1., options);
   WoodburyCorrection correction(hessian);
   NoInertiaCorrection regularization;
   for (size_t m: {0, 255, 0}) {
      const DenseTestModel padded_model = make_model(m);
      OptimizationProblem problem(padded_model, 100.);
      Subproblem subproblem(problem, hessian, regularization);
      const size_t dimension = subproblem.number_variables + subproblem.number_constraints;
      ScaledIdentitySolver solver(dimension);
      double scale = 1.;
      for (size_t rank: {0, 1, 2, 2, 0, 2, 1}) {
         scale += 0.25;
         solver.diagonal = 1. + scale / 10.;
         hessian.set_columns(rank, scale);
         correction.update(subproblem, solver);
         Vector<double> solution(dimension, 1. / solver.diagonal);
         correction.apply(solution);
         const double squared_scale = scale * scale;
         const double a = solver.diagonal + (0 < rank ? squared_scale : 0.) + (1 < rank ? squared_scale / 16. : 0.);
         const double b = (0 < rank ? squared_scale / 2. : 0.) + (1 < rank ? squared_scale / 4. : 0.);
         const double c = solver.diagonal + (0 < rank ? squared_scale / 4. : 0.) + (1 < rank ? squared_scale : 0.);
         const double determinant = a * c - b * b;
         EXPECT_NEAR(solution[0], (c - b) / determinant, 1e-14);
         EXPECT_NEAR(solution[1], (a - b) / determinant, 1e-14);
         for (size_t i = 2; i < dimension; ++i) {
            EXPECT_NEAR(solution[i], 1. / solver.diagonal, 1e-14);
         }
      }
   }
}
