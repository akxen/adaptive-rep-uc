"""Class implementing Model Predictive Controller to calibrate emissions intensity baseline"""

import os
import pickle

import pandas as pd
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from common import CommonComponents


class MPCController:
    def __init__(self, output_dir=os.path.join(os.path.dirname(__name__), os.path.pardir, 'output')):
        self.output_dir = output_dir

        # Object containing model data
        self.data = ModelData()

        # Components common to both UC and MPC models
        self.common = CommonComponents()

        # Solver options
        self.keepfiles = True
        self.solver_options = {}
        self.opt = SolverFactory('cplex', solver_io='lp')

    def define_sets(self, m):
        """Define model sets"""

        # Generators eligible to receive subsidy
        m.G = Set(initialize=self.data.scheduled_duids)

        # Calibration intervals
        m.C = RangeSet(1, 6, ordered=True)

        return m

    @staticmethod
    def define_parameters(m):
        """Define model parameters"""

        # Generator energy output - forecast over calibration interval
        m.ENERGY_FORECAST = Param(m.G, m.C, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(initialize=0, mutable=True)

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

        def generator_revenue_rule(_m, g, c):
            """Revenue obtained from generator for a given scenario realisation"""

            return (m.EMISSIONS_RATE[g] - m.baseline[c]) * m.ENERGY_FORECAST[g, c] * m.PERMIT_PRICE

        # Generator revenue from a given scenario realisation
        m.GENERATOR_REVENUE = Expression(m.G, m.C, rule=generator_revenue_rule)

        def interval_revenue_rule(_m, c):
            """Total forecast revenue in each calibration interval"""

            return sum(m.GENERATOR_REVENUE[g, c] for g in m.G)

        # Interval revenue
        m.INTERVAL_REVENUE = Expression(m.C, rule=interval_revenue_rule)

        def cumulative_revenue_rule(_m, c):
            """Cumulative scheme revenue"""

            return sum(m.INTERVAL_REVENUE[i] for i in m.C if i <= c)

        # Cumulative scheme revenue to be accrued over calibration horizon
        m.CUMULATIVE_REVENUE = Expression(m.C, rule=cumulative_revenue_rule)

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
            revenue = sum(m.GENERATOR_REVENUE[g, c] for g in m.G for c in m.C)

            return m.REVENUE_START + revenue == m.REVENUE_TARGET

        # Revenue target
        m.REVENUE_TARGET_CONS = Constraint(rule=revenue_target_rule)

        def revenue_lower_bound_rule(_m, i):
            """Ensure revenue does not breach a lower bound over any calibration interval"""

            return m.REVENUE_START + sum(m.INTERVAL_REVENUE[c] for c in m.C if c <= i) >= m.REVENUE_FLOOR

        # Ensure revenue never goes below floor
        m.REVENUE_FLOOR_CONS = Constraint(m.C, rule=revenue_lower_bound_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function. Want to minimise changes to baseline over successive intervals."""

        # Minimise squared difference between baseline in successive intervals
        m.OBJECTIVE = Objective(expr=m.BASELINE_DEVIATION)

        return m

    def construct_model(self):
        """Construct MPC model"""

        # Initialise model
        m = ConcreteModel()

        # Define sets
        m = self.define_sets(m)

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

    def update_parameters(self, m, year, week, baseline_start, revenue_start, revenue_target, revenue_floor,
                          permit_price):
        """Update MPC model parameters"""

        # Update forecast energy output
        generator_forecast = self.get_week_generator_energy_forecast(year, week, n_intervals=m.C.last())

        # Only retain keys for generators eligible for rebate / penalty
        forecast_update = {k: v for k, v in generator_forecast.items() if k in m.ENERGY_FORECAST.keys()}

        # Update parameter values
        m.ENERGY_FORECAST.store_values(forecast_update)

        # Update revenue at start of calibration interval
        m.REVENUE_START = float(revenue_start)

        # Update baseline in interval prior to model run (use last value)
        m.BASELINE_START = float(baseline_start)

        # Update revenue target at end of calibration interval
        m.REVENUE_TARGET = float(revenue_target)

        # Update revenue floor
        m.REVENUE_FLOOR = float(revenue_floor)

        # Update permit price
        m.PERMIT_PRICE = float(permit_price)

        return m

    def load_interval_results(self, year, week, day):
        """Load results for a given interval"""

        with open(os.path.join(self.output_dir, f'interval_{year}_{week}_{day}.pickle'), 'rb') as f:
            results = pickle.load(f)

        return results

    def get_generator_interval_results(self, var_id, year, week, day):
        """Get results relating to a given generator. Index should be (generator, hour)."""

        # Load results
        results = self.load_interval_results(year, week, day)

        # Convert to series
        s = pd.Series(results[(year, week, day)][var_id])

        # Rename axes
        df = s.rename_axis(['generator', 'hour']).rename(var_id)

        # Reset index, pivot and only take first 24 hours. Remaining sequence will overlap with next day.
        df = df.reset_index().pivot(index='hour', columns='generator', values=var_id).loc[1:24]

        # Add interval ID to results DataFrame
        df = pd.concat([df], keys=[(year, week, day)], names=['year', 'week', 'day'])

        return df

    def get_generator_week_energy_proportion(self, year, week):
        """Get proportion of energy output from each generator for a given week"""

        # Container of DataFrames
        dfs = []

        for day in range(1, 8):
            # Get generator energy output for a given day
            df = self.get_generator_interval_results('e', year, week, day)

            # Append to main container
            dfs.append(df)

        # Concatenate DataFrames
        df_c = pd.concat(dfs)

        # Total energy output over week
        total_energy_output = df_c.groupby('week').sum().sum(axis=1).values[0]

        # Proportion of energy delivered from each generator
        energy_output_proportion = df_c.sum().div(total_energy_output).to_dict()

        return energy_output_proportion

    def get_week_demand_forecast(self, year, week):
        """Simple demand forecast. Use value of demand for corresponding week of previous year"""

        # Demand in a given year
        demand = sum(sum(v.values()) for k, v in self.data.demand.items() if (k[0] == year-1) and (k[1] == week))

        return demand

    def get_week_generator_energy_forecast(self, year, week, n_intervals):
        """Based on proportion of energy output in last period, forecast generator energy output in future intervals"""

        # Get proportion of total output produced by each generator for a given week
        generation_prop = self.get_generator_week_energy_proportion(year, week)

        # Get demand forecast over the next 'n' intervals
        demand_forecast = {i: self.get_week_demand_forecast(year, week + i) for i in range(1, n_intervals + 1)}

        # Generator energy forecast over 'n' future intervals
        generator_forecast = {}

        for g in generation_prop.keys():
            for i in demand_forecast.keys():
                generator_forecast[(g, i)] = generation_prop[g] * demand_forecast[i]

        return generator_forecast

    def run_baseline_updater(self, m, year, week, baseline_start, revenue_start, revenue_target, revenue_floor,
                             permit_price):
        """Run model to update baseline"""

        # Update model parameters
        m = self.update_parameters(m, year, week, baseline_start, revenue_start, revenue_target, revenue_floor,
                                   permit_price)

        # Solve model
        m, status = self.solve_model(m)

        if status['Solver'][0]['Termination condition'].key != 'optimal':
            raise Exception(f'Failed to solve model. Year {year} week {week}: {status}')

        # Model parameters
        parameters = {'year': year, 'week': week, 'baseline_start': baseline_start, 'revenue_start': revenue_start,
                      'revenue_target': revenue_target, 'revenue_floor': revenue_floor, 'permit_price': permit_price}

        # MPC model results
        results = {'baseline_trajectory': m.baseline.get_values(),
                   'parameters': parameters,
                   'interval_revenue': {k: v.expr() for k, v in m.INTERVAL_REVENUE.items()},
                   'cumulative_revenue': {k: v.expr() for k, v in m.CUMULATIVE_REVENUE.items()}}

        return results

    def save_results(self, year, week, results):
        """Save MPC results"""

        with open(os.path.join(self.output_dir, f'mpc_{year}_{week}.pickle'), 'wb') as f:
            pickle.dump(results, f)

    def get_cumulative_scheme_revenue(self, year, week):
        """Get cumulative scheme revenue up until a given week (revenue at start of week)"""

        # All interval result files
        files = [f for f in os.listdir(self.output_dir) if ('interval' in f) and ('.pickle' in f)]

        # Cumulative scheme revenue
        cumulative_revenue = 0

        for f in files:
            y, w, d = int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[3].replace('.pickle', ''))

            if (y <= year) and (w < week):
                # Load interval results
                results = self.load_interval_results(y, w, d)

                # Update cumulative scheme revenue
                cumulative_revenue += results[(y, w, d)]['DAY_SCHEME_REVENUE']

        return cumulative_revenue


if __name__ == '__main__':
    mpc = MPCController()

    model_year, model_week, model_day = 2018, 2, 1
    # df_r = mpc.get_generator_interval_results('e', year, week, day)
    # energy_prop = mpc.get_generator_week_energy_proportion(2018, 2)
    # week_demand = mpc.get_week_demand_forecast(model_year, model_week)
    # gen_forecast = mpc.get_week_generator_energy_forecast(model_year, model_week, 6)

    # mpc_model = mpc.construct_model()
    # mpc_model = mpc.update_parameters(mpc_model, model_year, model_week, revenue_start=0, baseline_start=0.9,
    #                                   revenue_target=0, revenue_floor=0, permit_price=40)
    #
    # mpc_model, model_status = mpc.solve_model(mpc_model)
    # mpc_results = mpc.run_baseline_updater(mpc_model, model_year, model_week, baseline_start=0.9, revenue_start=0,
    #                                        revenue_target=0,  revenue_floor=0, permit_price=40)

    cumulative_scheme_revenue = mpc.get_cumulative_scheme_revenue(2018, 3)
