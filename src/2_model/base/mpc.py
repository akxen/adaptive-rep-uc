"""Class implementing Model Predictive Controller to calibrate emissions intensity baseline"""

from pyomo.environ import *

class MPCController:
    def __init__(self):
        pass

    def define_sets(self, m):
        """Define model sets"""

        # Calibration intervals
        m.C = RangeSet(1, 6, ordered=True)

        # Scenarios per calibration interval
        m.S = RangeSet(1, 5, ordered=True)

        return m

    def define_parameters(self, m):
        """Define model parameters"""

        # Generator energy output
        m.ENERGY = Param(m.G, m.C, m.S, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(m.C, initialize=0, mutable=True)

        # Scheme revenue for first calibration interval
        m.REVENUE_START = Param(initialize=0, mutable=True)

        # Revenue target at end of final calibration interval
        m.REVENUE_TARGET = Param(initialize=0, mutable=True)

        # Minimum cumulative scheme revenue for any calibration interval
        m.REVENUE_FLOOR = Param(initialize=0, mutable=True)

        # Baseline in preceding calibration interval
        m.INITIAL_BASELINE = Param(initialize=0, mutable=True)

        # Probability associated with scenario realisation
        m.PROBABILITY = Param(m.S, initialize=0, mutable=True)

        return m

    def define_variables(self, m):
        """Define model variables"""

        # Emissions intensity baseline (tCO2/MWh)
        m.baseline = Var(m.C, initialize=0)

    def define_expressions(self, m):
        """Define model expressions"""

        def scenario_generator_revenue_rule(_m, g, c, s):
            """Revenue obtained from generator for a given scenario realisation"""

            return (m.E[g] - m.baseline[c]) * m.GAMMA[g, c, s] * m.PERMIT_PRICE[c]

        # Generator revenue from a given scenario realisation
        m.GENERATOR_REVENUE = Expression(m.G, m.C, m.S, rule=scenario_generator_revenue_rule)

        return m

    def define_constraints(self, m):
        """Define model constraints"""

        def revenue_target_rule(_m):
            """Ensure expected scheme revenue = target at end of calibration interval"""

            # Expected revenue over all calibration intervals
            revenue = sum(m.PROBABILITY[s] * m.GENERATOR_REVENUE[g, c, s] for g in m.G for c in m.C for s in m.S)

            return m.REVENUE_START + revenue == m.REVENUE_TARGET

        # Revenue target
        m.REVENUE_TARGET_CONS = Constraint(rule=revenue_target_rule)

        def revenue_lower_bound_rule(_m, i):
            """Ensure revenue does not breach a lower bound over any calibration interval"""



    def construct_model(self):
        """Construct MPC model"""
        pass

    def solve_model(self):
        """Solve MPC model"""
        pass

    def update_parameters(self, m):
        """Update MPC model parameters"""
        pass
