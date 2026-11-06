// Copyright (c) 2026 Charlie Vanaret
// Licensed under the MIT license. See LICENSE file in the project directory for details.

#include <cmath>
#include <gtest/gtest.h>
#include "../DenseTestModel.hpp"
#include "Uno.hpp"
#include "ingredients/inequality_handling_methods/InequalityHandlingMethod.hpp"
#include "ingredients/inequality_handling_methods/InequalityHandlingMethodFactory.hpp"
#include "ingredients/subproblem_solvers/SymmetricIndefiniteLinearSolverFactory.hpp"
#include "optimization/OptimizationProblem.hpp"
#include "optimization/Result.hpp"
#include "options/DefaultOptions.hpp"
#include "options/Options.hpp"
#include "options/Presets.hpp"
#include "tools/Logger.hpp"

using namespace uno;

TEST(SecondOrderCorrections, UnconstrainedLineSearchDoesNotRepeatObjectiveEvaluations) {
   if (SymmetricIndefiniteLinearSolverFactory::available_solvers().empty()) {
      GTEST_SKIP() << "requires a direct linear solver";
   }
   DenseTestModel model("rosenbrock", 2, 0, ProblemType::NONLINEAR,
      [](const Vector<double>& x) { return std::pow(1. - x[0], 2) + 100. * std::pow(x[1] - x[0] * x[0], 2); },
      [](const Vector<double>& x, Vector<double>& g) {
         g[0] = 2. * (x[0] - 1.) - 400. * x[0] * (x[1] - x[0] * x[0]);
         g[1] = 200. * (x[1] - x[0] * x[0]);
      },
      [](const Vector<double>&, Vector<double>&) { },
      [](const Vector<double>&, double*) { },
      [](const Vector<double>& x, double w, const Vector<double>&, double* h) {
         h[0] = w * (1200. * x[0] * x[0] - 400. * x[1] + 2.);
         h[1] = -400. * w * x[0];
         h[2] = 200. * w;
      }, {-Inf, -Inf}, {Inf, Inf}, {}, {}, {-1.2, 1.});
   Options options;
   DefaultOptions::load(options);
   Presets::set(options, "ipopt");
   Logger::level = SILENT;
   Uno solver;
   const Result with_soc = solver.solve(model, options);
   options.set_integer("SOC_max_iterations", 0);
   const Result without_soc = solver.solve(model, options);
   ASSERT_EQ(with_soc.optimization_status, OptimizationStatus::SUCCESS);
   ASSERT_EQ(without_soc.optimization_status, OptimizationStatus::SUCCESS);
   EXPECT_EQ(with_soc.number_objective_evaluations, without_soc.number_objective_evaluations);
   EXPECT_EQ(with_soc.number_iterations, without_soc.number_iterations);
   EXPECT_DOUBLE_EQ(with_soc.primal_solution[0], without_soc.primal_solution[0]);
   EXPECT_DOUBLE_EQ(with_soc.primal_solution[1], without_soc.primal_solution[1]);
}

TEST(SecondOrderCorrections, ConstraintRowsIncludingFixedVariablesKeepCorrectionsAvailable) {
   if (SymmetricIndefiniteLinearSolverFactory::available_solvers().empty()) {
      GTEST_SKIP() << "requires a direct linear solver";
   }
   for (const bool fixed_variable: {false, true}) {
      const size_t m = fixed_variable ? 0 : 1;
      DenseTestModel model("constraint_rows", 1, m, ProblemType::NONLINEAR,
         [](const Vector<double>& x) { return x[0] * x[0] / 2.; },
         [](const Vector<double>& x, Vector<double>& g) { g[0] = x[0]; },
         [fixed_variable](const Vector<double>& x, Vector<double>& c) { if (!fixed_variable) c[0] = x[0] * x[0]; },
         [fixed_variable](const Vector<double>& x, double* j) { if (!fixed_variable) j[0] = 2. * x[0]; },
         [](const Vector<double>&, double w, const Vector<double>& y, double* h) {
            h[0] = w - (y.size() == 0 ? 0. : 2. * y[0]);
         }, {fixed_variable ? 1. : -Inf}, {fixed_variable ? 1. : Inf},
         std::vector<double>(m, 1.), std::vector<double>(m, 1.), {1.});
      Options options;
      DefaultOptions::load(options);
      Presets::set(options, "ipopt");
      const OptimizationProblem problem(model, options.get_double("residual_scaling_threshold"));
      std::vector<OptionOverride> overrides;
      auto method = InequalityHandlingMethodFactory::create(problem, false, 1., options, overrides);
      method->initialize_memory();
      EXPECT_TRUE(method->has_second_order_corrections()) << "fixed variable: " << fixed_variable;
   }
}
