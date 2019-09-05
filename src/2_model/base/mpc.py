"""Class implementing Model Predictive Controller to calibrate emissions intensity baseline"""

import os
import pickle

import pandas as pd
from pyomo.environ import *

from data import ModelData
from common import CommonComponents


class MPCController:
    def __init__(self, output_dir=os.path.join(os.path.dirname(__name__), os.path.pardir, 'output')):
        self.output_dir = output_dir

        # Object containing model data
        self.data = ModelData()

        # Components common to both UC and MPC models
        self.common = CommonComponents()

    @staticmethod
    def define_sets(m):
        """Define model sets"""

        # Calibration intervals
        m.C = RangeSet(1, 6, ordered=True)

        return m

    @staticmethod
    def define_parameters(m):
        """Define model parameters"""

        # Generator energy output - forecast over calibration interval
        m.ENERGY_FORECAST = Param(m.G_SCHEDULED, m.C, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(m.C, initialize=0, mutable=True)

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

        def scenario_generator_revenue_rule(_m, g, c):
            """Revenue obtained from generator for a given scenario realisation"""

            return (m.EMISSIONS_RATE[g] - m.baseline[c]) * m.ENERGY_FORECAST[g, c] * m.PERMIT_PRICE[c]

        # Generator revenue from a given scenario realisation
        m.GENERATOR_REVENUE = Expression(m.G_SCHEDULED, m.C, rule=scenario_generator_revenue_rule)

        return m

    @staticmethod
    def define_constraints(m):
        """Define model constraints"""

        def revenue_target_rule(_m):
            """Ensure expected scheme revenue = target at end of calibration interval"""

            # Expected revenue over all calibration intervals
            revenue = sum(m.GENERATOR_REVENUE[g, c] for g in m.G_SCHEDULED for c in m.C)

            return m.REVENUE_START + revenue == m.REVENUE_TARGET

        # Revenue target
        m.REVENUE_TARGET_CONS = Constraint(rule=revenue_target_rule)

        def revenue_lower_bound_rule(_m, i):
            """Ensure revenue does not breach a lower bound over any calibration interval"""

            return (m.REVENUE_START + sum(m.GENERATOR_REVENUE[g, c] for g in m.G_SCHEDULED for c in m.C if c <= i)
                    >= m.REVENUE_FLOOR)

        # Ensure revenue never goes below floor
        m.REVENUE_FLOOR_CONS = Constraint(m.C, rule=revenue_lower_bound_rule)

        return m

    def construct_model(self):
        """Construct MPC model"""

        # Initialise model
        m = ConcreteModel()

        # Define sets common to both MPC and UC models
        m = self.common.define_sets(m)

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

        return m

    def solve_model(self):
        """Solve MPC model"""
        pass

    def update_parameters(self, m, year, week, revenue_start=0, baseline_start=0):
        """Update MPC model parameters"""

        # Update forecast energy output
        generator_forecast = self.get_week_generator_energy_forecast(year, week, n_intervals=m.C.last())

        # Update revenue at start of calibration interval
        m.REVENUE_START = float(revenue_start)

        # Update baseline in interval prior to model run (use last value)
        m.BASELINE_START = float(baseline_start)

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

        # Convert to DataFrame. Only take first 24 hours (rest is overlap into next day)
        df = (s.rename_axis(['generator', 'hour']).rename(var_id).reset_index()
                  .pivot(index='hour', columns='generator', values=var_id).loc[1:24])

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


if __name__ == '__main__':
    mpc = MPCController()

    model_year, model_week, model_day = 2018, 2, 1
    # df_r = mpc.get_generator_interval_results('e', year, week, day)
    # energy_prop = mpc.get_generator_week_energy_proportion(2018, 2)
    # week_demand = mpc.get_week_demand_forecast(model_year, model_week)
    # gen_forecast = mpc.get_week_generator_energy_forecast(model_year, model_week, 6)

    mpc_model = mpc.construct_model()
