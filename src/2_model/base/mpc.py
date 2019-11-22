"""Class implementing Model Predictive Controller to calibrate emissions intensity baseline"""

import os
import pickle

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from common import CommonComponents
from analysis import AnalyseResults


class MPCController:
    def __init__(self, output_dir=os.path.join(os.path.dirname(__name__), os.path.pardir, 'output')):
        self.output_dir = output_dir

        # Object containing model data
        self.data = ModelData()

        # Components common to both UC and MPC models
        self.common = CommonComponents()

        # Class used to analyse model results
        self.analysis = AnalyseResults()

        # Solver options
        self.keepfiles = True
        self.solver_options = {}
        self.opt = SolverFactory('cplex', solver_io='lp')

    def define_sets(self, m, generators, n_intervals, n_scenarios):
        """Define model sets"""

        # Generators eligible to receive subsidy
        m.G = Set(initialize=generators)

        # Calibration intervals
        m.C = RangeSet(1, n_intervals, ordered=True)

        # Scenarios
        m.S = RangeSet(1, n_scenarios, ordered=True)

        return m

    @staticmethod
    def define_parameters(m):
        """Define model parameters"""

        # Generator energy output - forecast over calibration interval
        m.ENERGY_FORECAST = Param(m.G, m.C, m.S, initialize=0, mutable=True)

        # Scenario probability
        m.SCENARIO_PROBABILITY = Param(m.G, m.C, m.S, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(m.G, initialize=0, mutable=True)

        # Scheme revenue for first calibration interval
        m.REVENUE_START = Param(initialize=0, mutable=True)

        # Revenue target at end of final calibration interval
        m.REVENUE_TARGET = Param(initialize=0, mutable=True)

        # Minimum cumulative scheme revenue for any calibration interval
        m.REVENUE_FLOOR = Param(initialize=0, mutable=True)

        # Baseline in preceding calibration interval
        m.BASELINE_START = Param(initialize=0, mutable=True)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables"""

        # Emissions intensity baseline (tCO2/MWh)
        m.baseline = Var(m.C, within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define model expressions"""

        def generator_scenario_revenue_rule(_m, g, c, s):
            """Revenue obtained from generator for a given scenario realisation"""

            return (m.EMISSIONS_RATE[g] - m.baseline[c]) * m.ENERGY_FORECAST[g, c, s] * m.PERMIT_PRICE[g]

        # Generator revenue from a given scenario realisation
        m.GENERATOR_SCENARIO_REVENUE = Expression(m.G, m.C, m.S, rule=generator_scenario_revenue_rule)

        def generator_expected_revenue_rule(_m, g, c):
            """Expected revenue over all scenarios for a given calibration interval"""

            return sum(m.GENERATOR_SCENARIO_REVENUE[g, c, s] * m.SCENARIO_PROBABILITY[g, c, s] for s in m.S)

        # Generator expected revenue
        m.GENERATOR_EXPECTED_REVENUE = Expression(m.G, m.C, rule=generator_expected_revenue_rule)

        def interval_expected_revenue_rule(_m, c):
            """Total forecast revenue in each calibration interval"""

            return sum(m.GENERATOR_EXPECTED_REVENUE[g, c] for g in m.G)

        # Interval revenue
        m.INTERVAL_EXPECTED_REVENUE = Expression(m.C, rule=interval_expected_revenue_rule)

        def cumulative_expected_revenue_rule(_m, c):
            """Cumulative scheme revenue"""

            return sum(m.INTERVAL_EXPECTED_REVENUE[i] for i in m.C if i <= c)

        # Cumulative scheme revenue to be accrued over calibration horizon
        m.CUMULATIVE_EXPECTED_REVENUE = Expression(m.C, rule=cumulative_expected_revenue_rule)

        def baseline_deviation_rule(_m):
            """Squared baseline deviation between successive intervals"""

            # Deviation for first period
            first_period = (m.BASELINE_START - m.baseline[m.C.first()]) * (m.BASELINE_START - m.baseline[m.C.first()])

            # Deviation for remaining periods
            remaining_periods = sum((m.baseline[c-1] - m.baseline[c]) * (m.baseline[c-1] - m.baseline[c])
                                    for c in m.C if c != m.C.first())

            return first_period + remaining_periods

        # Squared baseline deviation
        m.BASELINE_DEVIATION = Expression(rule=baseline_deviation_rule)

        return m

    @staticmethod
    def define_constraints(m):
        """Define model constraints"""

        def revenue_target_rule(_m):
            """Ensure expected scheme revenue = target at end of calibration interval"""

            # Expected revenue over all calibration intervals
            return m.REVENUE_START + sum(m.INTERVAL_EXPECTED_REVENUE[c] for c in m.C) == m.REVENUE_TARGET

        # Revenue target
        m.REVENUE_TARGET_CONS = Constraint(rule=revenue_target_rule)

        def revenue_floor_rule(_m, c):
            """Revenue floor"""

            return m.REVENUE_START + sum(m.INTERVAL_EXPECTED_REVENUE[i] for i in m.C if i <= c) >= m.REVENUE_FLOOR

        # Revenue floor
        m.REVENUE_FLOOR_CONS = Constraint(m.C, rule=revenue_floor_rule)
        m.REVENUE_FLOOR_CONS.deactivate()

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function. Want to minimise changes to baseline over successive intervals."""

        # Minimise squared difference between baseline in successive intervals
        m.OBJECTIVE = Objective(expr=m.BASELINE_DEVIATION, sense=minimize)

        return m

    def construct_model(self, generators, n_intervals, n_scenarios):
        """Construct MPC model"""

        # Initialise model
        m = ConcreteModel()

        # Define sets
        m = self.define_sets(m, generators, n_intervals, n_scenarios)

        # Define parameters common to both MPC and UC models
        m = self.common.define_parameters(m)

        # Define model parameters
        m = self.define_parameters(m)

        # Define variables
        m = self.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective function
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status

    @staticmethod
    def update_parameters(m, baseline_start, revenue_start, revenue_target, revenue_floor,
                          permit_price, energy_forecast, scenario_probabilities):
        """Update MPC model parameters"""

        # Update baseline in interval prior to model run (use last value)
        m.BASELINE_START = float(baseline_start)

        # Update revenue at start of calibration interval
        m.REVENUE_START = float(revenue_start)

        # Update revenue target at end of calibration interval
        m.REVENUE_TARGET = float(revenue_target)

        # Update revenue floor if value provided
        if revenue_floor is not None:
            m.REVENUE_FLOOR = float(revenue_floor)

        # Update permit price
        m.PERMIT_PRICE.store_values(permit_price)

        # Update parameter values
        m.ENERGY_FORECAST.store_values(energy_forecast)

        # Update probabilities for each scenario
        m.SCENARIO_PROBABILITY.store_values(scenario_probabilities)

        return m

    def run_baseline_updater(self, m, year, week, baseline_start, revenue_start, revenue_target, revenue_floor,
                             permit_price, energy_forecast, scenario_probabilities):
        """Run model to update baseline"""

        # Update model parameters
        m = self.update_parameters(m, baseline_start, revenue_start, revenue_target, revenue_floor, permit_price,
                                   energy_forecast, scenario_probabilities)

        # Solve model
        m, status = self.solve_model(m)

        if status['Solver'][0]['Termination condition'].key != 'optimal':
            raise Exception(f'Failed to solve model. Year {year} week {week}: {status}')

        # Model parameters
        parameters = {'year': year, 'week': week,
                      'baseline_start': m.BASELINE_START.value,
                      'revenue_start': m.REVENUE_START.value,
                      'revenue_target': m.REVENUE_TARGET.value,
                      'revenue_floor': m.REVENUE_FLOOR.value}

        # MPC model results
        results = {'baseline_trajectory': m.baseline.get_values(),
                   'parameters': parameters,
                   'interval_revenue': {k: v.expr() for k, v in m.INTERVAL_EXPECTED_REVENUE.items()},
                   'cumulative_revenue': {k: v.expr() for k, v in m.CUMULATIVE_EXPECTED_REVENUE.items()},
                   'energy_forecast': {k: v.value for k, v in m.ENERGY_FORECAST.items()},
                   'scenario_probability': {k: v.value for k, v in m.SCENARIO_PROBABILITY.items()},
                   'permit_price': {k: v.value for k, v in m.PERMIT_PRICE.items()}}

        return results

    @staticmethod
    def save_results(year, week, results, output_dir):
        """Save MPC results"""

        with open(os.path.join(output_dir, f'mpc_{year}_{week}.pickle'), 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    mpc = MPCController()

    model_year, model_week, model_day = 2018, 2, 1

    model = mpc.construct_model(mpc.data.generators.index, n_intervals=3, n_scenarios=5)
