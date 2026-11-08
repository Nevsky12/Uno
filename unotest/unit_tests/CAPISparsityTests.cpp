// Copyright (c) 2026 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>
#include "../../interfaces/C/Uno_C_API.h"
#include "ingredients/subproblem_solvers/SymmetricIndefiniteLinearSolverFactory.hpp"
#include "tools/Logger.hpp"

using namespace uno;

namespace {
   class CAPISparsity: public testing::TestWithParam<uno_int> {
   protected:
      void* model{nullptr};
      uno_int base{0};

      void SetUp() override {
         Logger::level = SILENT;
         this->base = GetParam();
         this->model = uno_create_unconstrained_model(UNO_PROBLEM_NONLINEAR, 2, this->base);
         ASSERT_NE(this->model, nullptr);
      }

      void TearDown() override {
         uno_destroy_model(this->model);
      }
   };
} // namespace

TEST_P(CAPISparsity, RejectsIndicesOutsideMatrixDimensions) {
   const uno_int first = this->base;
   const uno_int past_constraint = this->base + 1;
   const uno_int past_variable = this->base + 2;
   EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, &past_constraint, &first, nullptr));
   EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, &first, &past_variable, nullptr));
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, &past_variable, &first, nullptr));
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_UPPER_TRIANGLE, &first, &past_variable, nullptr));
}

TEST_P(CAPISparsity, RejectsExtremeIndicesBeforeSubtractingBase) {
   const uno_int first = this->base;
   for (const uno_int invalid: {this->base - 1, std::numeric_limits<uno_int>::min(), std::numeric_limits<uno_int>::max()}) {
      EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, &invalid, &first, nullptr));
      EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, &first, &invalid, nullptr));
      EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, &invalid, &first, nullptr));
      EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, &first, &invalid, nullptr));
   }
}

TEST_P(CAPISparsity, RejectsNegativeNonzeroCountsWithoutThrowing) {
   EXPECT_NO_THROW({
      EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, -1, nullptr, nullptr, nullptr));
   });
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, -1, UNO_LOWER_TRIANGLE, nullptr, nullptr, nullptr));
}

TEST_P(CAPISparsity, RejectsExtremeComponentIndicesBeforeSubtractingBase) {
   ASSERT_TRUE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr));
   for (const uno_int invalid: {std::numeric_limits<uno_int>::min(), std::numeric_limits<uno_int>::max()}) {
      EXPECT_FALSE(uno_set_variable_lower_bound(this->model, invalid, 0.));
      EXPECT_FALSE(uno_set_variable_upper_bound(this->model, invalid, 0.));
      EXPECT_FALSE(uno_set_constraint_lower_bound(this->model, invalid, 0.));
      EXPECT_FALSE(uno_set_constraint_upper_bound(this->model, invalid, 0.));
      EXPECT_FALSE(uno_set_initial_primal_iterate_component(this->model, invalid, 0.));
      EXPECT_FALSE(uno_set_initial_dual_iterate_component(this->model, invalid, 0.));
   }
   EXPECT_TRUE(uno_set_variable_lower_bound(this->model, this->base + 1, -1.));
   EXPECT_TRUE(uno_set_variable_upper_bound(this->model, this->base + 1, 1.));
   EXPECT_TRUE(uno_set_constraint_lower_bound(this->model, this->base, -1.));
   EXPECT_TRUE(uno_set_constraint_upper_bound(this->model, this->base, 1.));
   EXPECT_TRUE(uno_set_initial_primal_iterate_component(this->model, this->base + 1, 0.));
   EXPECT_TRUE(uno_set_initial_dual_iterate_component(this->model, this->base, 0.));
}

TEST_P(CAPISparsity, RequiresIndexArraysOnlyForNonemptyPatterns) {
   const uno_int first = this->base;
   EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, nullptr, &first, nullptr));
   EXPECT_FALSE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 1, &first, nullptr, nullptr));
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, nullptr, &first, nullptr));
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, &first, nullptr, nullptr));
   EXPECT_TRUE(uno_set_constraints(this->model, 1, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr));
   EXPECT_TRUE(uno_set_lagrangian_hessian(this->model, 0, UNO_LOWER_TRIANGLE, nullptr, nullptr, nullptr));
}

TEST_P(CAPISparsity, RejectedPatternsPreserveThePreviousModel) {
   if (SymmetricIndefiniteLinearSolverFactory::available_solvers().empty()) {
      GTEST_SKIP() << "requires a direct linear solver";
   }
   const uno_int jacobian_rows[] = {this->base, this->base};
   const uno_int columns[] = {this->base, this->base + 1};
   const double constraint_bound[] = {3.};
   ASSERT_TRUE(uno_set_objective(this->model, UNO_MINIMIZE,
      [](uno_int, const double* x, double* f, void*) -> uno_int {
         *f = (x[0] - 1.) * (x[0] - 1.) + (x[1] - 2.) * (x[1] - 2.); return 0;
      },
      [](uno_int, const double* x, double* g, void*) -> uno_int {
         g[0] = 2. * (x[0] - 1.); g[1] = 2. * (x[1] - 2.); return 0;
      }));
   ASSERT_TRUE(uno_set_constraints(this->model, 1,
      [](uno_int, uno_int, const double* x, double* c, void*) -> uno_int { c[0] = x[0] + x[1]; return 0; },
      constraint_bound, constraint_bound, 2, jacobian_rows, columns,
      [](uno_int, uno_int, const double*, double* j, void*) -> uno_int { j[0] = 1.; j[1] = 1.; return 0; }));
   ASSERT_TRUE(uno_set_lagrangian_hessian(this->model, 2, UNO_LOWER_TRIANGLE, columns, columns,
      [](uno_int, uno_int, uno_int, const double*, double w, const double*, double* h, void*) -> uno_int {
         h[0] = 2. * w; h[1] = 2. * w; return 0;
      }));
   const uno_int invalid = this->base + 2;
   EXPECT_FALSE(uno_set_constraints(this->model, 2, nullptr, nullptr, nullptr, 1, &invalid, &this->base, nullptr));
   EXPECT_FALSE(uno_set_constraint_lower_bound(this->model, this->base + 1, 0.));
   EXPECT_FALSE(uno_set_lagrangian_hessian(this->model, 1, UNO_LOWER_TRIANGLE, &invalid, &this->base, nullptr));

   void* solver = uno_create_solver();
   ASSERT_NE(solver, nullptr);
   uno_set_solver_preset(solver, "ipopt");
   uno_set_solver_string_option(solver, "logger", "SILENT");
   uno_optimize(solver, this->model);
   EXPECT_EQ(uno_get_optimization_status(solver), UNO_SUCCESS);
   EXPECT_EQ(uno_get_solution_status(solver), UNO_FEASIBLE_KKT_POINT);
   EXPECT_NEAR(uno_get_primal_solution_component(solver, this->base), 1., 1e-7);
   EXPECT_NEAR(uno_get_primal_solution_component(solver, this->base + 1), 2., 1e-7);
   for (const uno_int invalid: {std::numeric_limits<uno_int>::min(), std::numeric_limits<uno_int>::max()}) {
      EXPECT_THROW(uno_get_primal_solution_component(solver, invalid), std::runtime_error);
      EXPECT_THROW(uno_get_constraint_dual_solution_component(solver, invalid), std::runtime_error);
      EXPECT_THROW(uno_get_lower_bound_dual_solution_component(solver, invalid), std::runtime_error);
      EXPECT_THROW(uno_get_upper_bound_dual_solution_component(solver, invalid), std::runtime_error);
   }
   uno_destroy_solver(solver);
}

INSTANTIATE_TEST_SUITE_P(Indexing, CAPISparsity, testing::Values(UNO_ZERO_BASED_INDEXING, UNO_ONE_BASED_INDEXING));
