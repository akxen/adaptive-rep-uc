"""Unit commitment model"""

import os
import pickle
from collections import OrderedDict

import pandas as pd
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from mpc import MPCController
from data import ModelData
from common import CommonComponents


class UnitCommitment:
    def __init__(self):
        # Pre-processed data for model construction
        self.data = ModelData()

        # Common model components
        self.common = CommonComponents()

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'mip tolerances integrality': 0.01
        self.opt = SolverFactory('cplex', solver_io='lp')

    def define_sets(self, m, overlap):
        """Define sets to be used in model"""

        # NEM regions
        m.R = Set(initialize=self.data.nem_regions)

        # NEM zones
        m.Z = Set(initialize=self.data.nem_zones)

        # Links between NEM zones
        m.L = Set(initialize=self.data.links)

        # Interconnectors for which flow limits are defined
        m.L_I = Set(initialize=self.data.links_constrained)

        # Scheduled generators
        m.G_SCHEDULED = Set(initialize=self.data.scheduled_duids)

        # Semi-scheduled generators (e.g. wind, solar)
        m.G_SEMI_SCHEDULED = Set(initialize=self.data.semi_scheduled_duids)

        # Existing storage units
        m.G_STORAGE = Set(initialize=self.data.storage_duids)

        # Generators considered in analysis - only semi and semi-scheduled generators (and storage units)
        m.G_MODELLED = m.G_SCHEDULED.union(m.G_SEMI_SCHEDULED).union(m.G_STORAGE)

        # Thermal units
        m.G_THERM = Set(initialize=self.data.get_thermal_unit_duids()).intersection(m.G_MODELLED)

        # Wind units
        m.G_WIND = Set(initialize=self.data.get_wind_unit_duids()).intersection(m.G_MODELLED)

        # Solar units
        m.G_SOLAR = Set(initialize=self.data.get_solar_unit_duids()).intersection(m.G_MODELLED)

        # Existing hydro units
        m.G_HYDRO = Set(initialize=self.data.get_hydro_unit_duids()).intersection(m.G_MODELLED)

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self.data.slow_start_duids).intersection(m.G_MODELLED)

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self.data.quick_start_duids).intersection(m.G_MODELLED)

        # All generators (storage units excluded)
        m.G = m.G_THERM.union(m.G_WIND).union(m.G_SOLAR).union(m.G_HYDRO)

        # Operating scenario hour
        m.T = RangeSet(1, 24 + overlap, ordered=True)

        return m

    def define_parameters(self, m):
        """Define unit commitment problem parameters"""

        def minimum_region_up_reserve_rule(_m, r):
            """Minimum upward reserve rule"""

            return float(self.data.minimum_reserve_levels[r])

        # Minimum up reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def ramp_rate_startup_rule(_m, g):
            """Startup ramp-rate (MW)"""

            return float(self.data.generators.loc[g, 'RR_STARTUP'])

        # Startup ramp-rate for existing and candidate thermal generators
        m.RR_SU = Param(m.G_THERM, rule=ramp_rate_startup_rule)

        def ramp_rate_shutdown_rule(_m, g):
            """Shutdown ramp-rate (MW)"""

            return float(self.data.generators.loc[g, 'RR_SHUTDOWN'])

        # Shutdown ramp-rate for existing and candidate thermal generators
        m.RR_SD = Param(m.G_THERM, rule=ramp_rate_shutdown_rule)

        def ramp_rate_normal_up_rule(_m, g):
            """Ramp-rate up (MW/h) - when running"""

            return float(self.data.generators.loc[g, 'RR_UP'])

        # Ramp-rate up (normal operation)
        m.RR_UP = Param(m.G_THERM, rule=ramp_rate_normal_up_rule)

        def ramp_rate_normal_down_rule(_m, g):
            """Ramp-rate down (MW/h) - when running"""

            return float(self.data.generators.loc[g, 'RR_DOWN'])

        # Ramp-rate down (normal operation)
        m.RR_DOWN = Param(m.G_THERM, rule=ramp_rate_normal_down_rule)

        def min_power_output_rule(_m, g):
            """Minimum power output for thermal generators"""

            return float(self.data.generators.loc[g, 'MIN_GEN'])

        # Minimum power output
        m.P_MIN = Param(m.G_THERM, rule=min_power_output_rule)

        def network_incidence_matrix_rule(_m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            return float(self.data.network_incidence_matrix.loc[l, z])

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def powerflow_min_rule(_m, l):
            """Minimum powerflow over network link"""

            return float(-self.data.powerflow_limits[l]['reverse'])

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L_I, rule=powerflow_min_rule)

        def powerflow_max_rule(_m, l):
            """Maximum powerflow over network link"""

            return float(self.data.powerflow_limits[l]['forward'])

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L_I, rule=powerflow_max_rule)

        def battery_efficiency_rule(_m, g):
            """Battery efficiency"""

            return float(self.data.storage.loc[g, 'EFFICIENCY'])

        # Battery efficiency
        m.BATTERY_EFFICIENCY = Param(m.G_STORAGE, rule=battery_efficiency_rule)

        # Lower bound for energy in storage unit at end of interval (assume = 0)
        m.Q_INTERVAL_END_LB = Param(m.G_STORAGE, initialize=0)

        def storage_unit_interval_end_ub_rule(_m, g):
            """
            Max energy in storage unit at end of interval

            Assume energy capacity (MWh) = registered capacity (MW). I.e unit can completely discharge within 1hr.
            """

            return float(self.data.storage.loc[g, 'REG_CAP'])

        # Upper bound for energy in storage unit at end of interval (assume = P_MAX)
        m.Q_INTERVAL_END_UB = Param(m.G_STORAGE, initialize=storage_unit_interval_end_ub_rule)

        # Energy in battery in interval prior to model start (assume battery initially completely discharged)
        m.Q0 = Param(m.G_STORAGE, initialize=0, mutable=True)

        def marginal_cost_rule(_m, g):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be updated for each year in model horizon.
            """

            if g in m.G:
                marginal_cost = float(self.data.generators.loc[g, 'SRMC_2016-17'])

            elif g in m.G_STORAGE:
                marginal_cost = float(self.data.storage.loc[g, 'SRMC_2016-17'])

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return marginal_cost

        # Marginal costs for all generators and time periods
        m.C_MC = Param(m.G.union(m.G_STORAGE), rule=marginal_cost_rule)

        def startup_cost_rule(_m, g):
            """
            Startup costs for existing and candidate thermal units
            """

            startup_cost = float(self.data.generators.loc[g, 'SU_COST_WARM'])

            # Shutdown cost cannot be negative
            assert startup_cost >= 0, 'Negative startup cost'

            return startup_cost

        def max_power_output_rule(_m, g):
            """Max power output for generators. Max discharging or charging power for storage units"""

            if g in m.G:
                return float(self.data.generators.loc[g, 'REG_CAP'])
            elif g in m.G_STORAGE:
                return float(self.data.storage.loc[g, 'REG_CAP'])
            else:
                raise Exception(f'Unexpected generator DUID: {g}')

        # Max power output
        m.P_MAX = Param(m.G.union(m.G_STORAGE), rule=max_power_output_rule)

        def max_storage_unit_energy_rule(_m, g):
            """Max energy storage capability. Assume capacity (MWh) = registered capacity (MW)"""

            return float(self.data.storage.loc[g, 'REG_CAP'])

        # Max energy capacity
        m.Q_MAX = Param(m.G_STORAGE, rule=max_storage_unit_energy_rule)

        # Generator startup costs - per MW
        m.C_SU = Param(m.G_THERM, rule=startup_cost_rule)

        # Generator shutdown costs - per MW - assume zero for now TODO: May need to update this assumption
        m.C_SD = Param(m.G_THERM, initialize=0)

        # Value of lost load [$/MWh]
        m.C_L = Param(initialize=10000)

        # Penalty for violating up reserve constraint
        m.C_UV = Param(initialize=1000)

        # Permit price - positive if generators eligible under scheme.
        m.PERMIT_PRICE = Param(m.G, initialize=0, mutable=True)

        # -------------------------------------------------------------------------------------------------------------
        # Parameters to update each time model is run
        # -------------------------------------------------------------------------------------------------------------
        # Baseline - defined for each dispatch interval (correct calculation of scheme revenue when update occurs)
        m.BASELINE = Param(m.T, initialize=0, mutable=True)

        # Initial on-state rule - must be updated each time model is run
        m.U0 = Param(m.G_THERM, within=Binary, mutable=True, initialize=1)

        # Power output in interval prior to model start (assume = 0 for now)
        m.P0 = Param(m.G.union(m.G_STORAGE), mutable=True, within=NonNegativeReals, initialize=0)

        # Wind output
        m.P_WIND = Param(m.G_WIND, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Solar output
        m.P_SOLAR = Param(m.G_SOLAR, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Hydro output
        m.P_HYDRO = Param(m.G_HYDRO, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Demand
        m.DEMAND = Param(m.Z, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_variables(m):
        """Define unit commitment problem variables"""

        # Upward reserve allocation [MW]
        m.r_up = Var(m.G_THERM.union(m.G_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Startup state variable
        m.v = Var(m.G_THERM, m.T, within=Binary, initialize=0)

        # On-state variable
        m.u = Var(m.G_THERM, m.T, within=Binary, initialize=1)

        # Shutdown state variable
        m.w = Var(m.G_THERM, m.T, within=Binary, initialize=0)

        # Power output above minimum dispatchable level
        m.p = Var(m.G_THERM, m.T, within=NonNegativeReals, initialize=0)

        # Total power output
        m.p_total = Var(m.G.difference(m.G_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Storage unit charging power
        m.p_in = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Storage unit discharging power
        m.p_out = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Storage unit energy (state of charge)
        m.q = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Powerflow between NEM zones
        m.p_flow = Var(m.L, m.T, initialize=0)

        # Lost-load
        m.p_V = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Up reserve constraint violation
        m.reserve_up_violation = Var(m.R, m.T, within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define unit commitment problem expressions"""

        def energy_output_rule(_m, g, t):
            """Energy output"""

            # If a storage unit
            if g in m.G_STORAGE:
                if t != m.T.first():
                    return (m.p_out[g, t - 1] + m.p_out[g, t]) / 2
                else:
                    return (m.P0[g] + m.p_out[g, t]) / 2

            # For all other units
            elif g in m.G.difference(m.G_STORAGE):
                if t != m.T.first():
                    return (m.p_total[g, t - 1] + m.p_total[g, t]) / 2
                else:
                    return (m.P0[g] + m.p_total[g, t]) / 2

            else:
                raise Exception(f'Unexpected unit encountered: {g}')

        # Energy output
        m.e = Expression(m.G.union(m.G_STORAGE), m.T, rule=energy_output_rule)

        def lost_load_energy_rule(_m, z, t):
            """
            Amount of lost-load energy.

            Note: Assumes lost-load energy in interval prior to model start (t=0) is zero.
            """

            if t != m.T.first():
                return (m.p_V[z, t] + m.p_V[z, t - 1]) / 2

            else:
                # Assumes no lost energy in interval preceding model start
                return m.p_V[z, t] / 2

        # Lost-load energy
        m.e_V = Expression(m.Z, m.T, rule=lost_load_energy_rule)

        def day_emissions_rule(_m):
            """Total emissions for a given interval"""

            return sum(m.e[g, t] * m.EMISSIONS_RATE[g] for g in m.G_THERM for t in range(1, 25))

        # Total scenario emissions
        m.DAY_EMISSIONS = Expression(rule=day_emissions_rule)

        def day_demand_rule(_m):
            """Total demand accounted for by given scenario"""

            return sum(m.DEMAND[z, t] for z in m.Z for t in range(1, 25))

        # Total scenario energy demand
        m.DAY_DEMAND = Expression(rule=day_demand_rule)

        def day_emissions_intensity_rule(_m):
            """Emissions intensity for a given interval"""

            return m.DAY_EMISSIONS / m.DAY_DEMAND

        # Emissions intensity for given day
        m.DAY_EMISSIONS_INTENSITY = Expression(rule=day_emissions_intensity_rule)

        def net_penalty_rule(_m, g, t):
            """Net penalty per MWh"""

            return (m.EMISSIONS_RATE[g] - m.BASELINE[t]) * m.PERMIT_PRICE[g]

        # Net penalty per MWh
        m.NET_PENALTY = Expression(m.G, m.T, rule=net_penalty_rule)

        def day_scheme_revenue_rule(_m):
            """Total scheme revenue accrued for a given day"""

            return sum(m.NET_PENALTY[g, t] * m.e[g, t] for g in m.G for t in range(1, 25))

        # Scheme revenue for a given day
        m.DAY_SCHEME_REVENUE = Expression(rule=day_scheme_revenue_rule)

        def thermal_operating_costs_rule(_m):
            """Cost to operate thermal generators for given scenario"""

            # Operating cost related to energy output + emissions charge
            operating_costs = sum((m.C_MC[g] + m.NET_PENALTY[g, t]) * m.e[g, t] for g in m.G_THERM for t in m.T)

            # Existing unit start-up and shutdown costs
            startup_shutdown_costs = (sum((m.C_SU[g] * m.v[g, t]) + (m.C_SD[g] * m.w[g, t])
                                          for g in m.G_THERM for t in m.T))

            # Total thermal unit costs
            total_cost = operating_costs + startup_shutdown_costs

            return total_cost

        # Operating cost - thermal units
        m.OP_T = Expression(rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(_m):
            """Cost to operate hydro generators"""

            return sum((m.C_MC[g] + m.NET_PENALTY[g, t]) * m.e[g, t] for g in m.G_HYDRO for t in m.T)

        # Operating cost - hydro generators
        m.OP_H = Expression(rule=hydro_operating_costs_rule)

        def wind_operating_costs_rule(_m):
            """Cost to operate wind generators"""

            # Existing wind generators - not eligible for subsidy
            total_cost = sum((m.C_MC[g] + m.NET_PENALTY[g, t]) * m.e[g, t] for g in m.G_WIND for t in m.T)

            return total_cost

        # Operating cost - wind units
        m.OP_W = Expression(rule=wind_operating_costs_rule)

        def solar_operating_costs_rule(_m):
            """Cost to operate solar generators"""

            # Existing solar generators - not eligible for subsidy
            total_cost = sum((m.C_MC[g] + m.NET_PENALTY[g, t]) * m.e[g, t] for g in m.G_SOLAR for t in m.T)

            return total_cost

        # Operating cost - solar units
        m.OP_S = Expression(rule=solar_operating_costs_rule)

        def storage_operating_costs_rule(_m):
            """Cost to operate storage units"""

            return sum(m.C_MC[g] * m.e[g, t] for g in m.G_STORAGE for t in m.T)

        # Operating cost - storage units
        m.OP_Q = Expression(rule=storage_operating_costs_rule)

        def lost_load_value_rule(_m):
            """Vale of lost load"""

            return sum(m.C_L * m.e_V[z, t] for z in m.Z for t in m.T)

        # Value of lost load
        m.OP_L = Expression(rule=lost_load_value_rule)

        def reserve_up_violation_value_rule(_m):
            """Value of up reserve constraint violation"""

            return sum(m.C_UV * m.reserve_up_violation[r, t] for r in m.R for t in m.T)

        # Up reserve violation penalty
        m.OP_U = Expression(rule=reserve_up_violation_value_rule)

        # Total operating cost for a given interval
        m.INTERVAL_COST = Expression(expr=m.OP_T + m.OP_H + m.OP_W + m.OP_S + m.OP_Q + m.OP_L + m.OP_U)

        # Objective function - sum of operational costs for a given interval
        m.OBJECTIVE_FUNCTION = Expression(expr=m.INTERVAL_COST)

        return m

    def define_constraints(self, m):
        """Define unit commitment problem constraints"""

        def reserve_up_rule(_m, r, t):
            """Ensure sufficient up power reserve in each region"""

            # Existing and candidate thermal gens + candidate storage units
            gens = m.G_THERM.union(m.G_STORAGE)

            # Subset of generators with NEM region
            gens_subset = [g for g in gens if self.data.duid_zone_map[g] in self.data.nem_region_zone_map[r]]

            return sum(m.r_up[g, t] for g in gens_subset) + m.reserve_up_violation[r, t] >= m.RESERVE_UP[r]

        # Upward power reserve rule for each NEM region
        m.RESERVE_UP_CONS = Constraint(m.R, m.T, rule=reserve_up_rule)

        def generator_state_logic_rule(_m, g, t):
            """
            Determine the operating state of the generator (startup, shutdown running, off)
            """

            if t == m.T.first():
                # Must use U0 if first period (otherwise index out of range)
                return m.u[g, t] - m.U0[g] == m.v[g, t] - m.w[g, t]

            else:
                # Otherwise operating state is coupled to previous period
                return m.u[g, t] - m.u[g, t - 1] == m.v[g, t] - m.w[g, t]

        # Unit operating state
        m.GENERATOR_STATE_LOGIC = Constraint(m.G_THERM, m.T, rule=generator_state_logic_rule)

        def minimum_on_time_rule(_m, g, t):
            """Minimum number of hours generator must be on"""

            if g in m.G_THERM:
                hours = self.data.generators.loc[g, 'MIN_ON_TIME']

            else:
                raise Exception(f'Min on time hours not found for generator: {g}')

            # Time index used in summation
            time_index = [k for k in range(t - int(hours) + 1, t + 1) if k >= 1]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum on time constraint
        m.MINIMUM_ON_TIME = Constraint(m.G_THERM, m.T, rule=minimum_on_time_rule)

        def minimum_off_time_rule(_m, g, t):
            """Minimum number of hours generator must be off"""

            if g in m.G_THERM:
                hours = self.data.generators.loc[g, 'MIN_OFF_TIME']

            else:
                raise Exception(f'Min off time hours not found for generator: {g}')

            # Time index used in summation
            time_index = [k for k in range(t - int(hours) + 1, t + 1) if k >= 1]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.w[g, j] for j in time_index) <= 1 - m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum off time constraint
        m.MINIMUM_OFF_TIME = Constraint(m.G_THERM, m.T, rule=minimum_off_time_rule)

        def ramp_rate_up_rule(_m, g, t):
            """Ramp-rate up constraint - normal operation"""

            # For all other intervals apart from the first
            if t != m.T.first():
                return (m.p[g, t] + m.r_up[g, t]) - m.p[g, t - 1] <= m.RR_UP[g]

            else:
                # Ramp-rate for first interval
                return m.p[g, t] + m.r_up[g, t] - m.P0[g] <= m.RR_UP[g]

        # Ramp-rate up limit
        m.RAMP_RATE_UP = Constraint(m.G_THERM, m.T, rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(_m, g, t):
            """Ramp-rate down constraint - normal operation"""

            # For all other intervals apart from the first
            if t != m.T.first():
                return - m.p[g, t] + m.p[g, t - 1] <= m.RR_DOWN[g]

            else:
                # Ramp-rate for first interval
                return - m.p[g, t] + m.P0[g] <= m.RR_DOWN[g]

        # Ramp-rate up limit
        m.RAMP_RATE_DOWN = Constraint(m.G_THERM, m.T, rule=ramp_rate_down_rule)

        def power_output_within_limits_rule(_m, g, t):
            """Ensure power output + reserves within capacity limits"""

            # Left hand-side of constraint
            lhs = m.p[g, t] + m.r_up[g, t]

            # Existing thermal units - fixed capacity
            if g in m.G_THERM:
                rhs_1 = (m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t]

                # If not the last period
                if t != m.T.last():
                    rhs_2 = (m.P_MAX[g] - m.RR_SD[g]) * m.w[g, t + 1]
                    rhs_3 = (m.RR_SU[g] - m.P_MIN[g]) * m.v[g, t + 1]

                    return lhs <= rhs_1 - rhs_2 + rhs_3

                # If the last period - startup and shutdown state variables assumed = 0
                else:
                    return lhs <= rhs_1

            else:
                raise Exception(f'Unknown generator: {g}')

        # Power output and reserves within limits
        m.POWER_OUTPUT_WITHIN_LIMITS = Constraint(m.G_THERM, m.T, rule=power_output_within_limits_rule)

        def total_power_thermal_rule(_m, g, t):
            """Total power output for thermal generators"""

            # Quick-start thermal generators
            if g in m.G_THERM_QUICK:
                # If not the last index
                if t != m.T.last():
                    return m.p_total[g, t] == m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1]) + m.p[g, t]

                # If the last index assume shutdown and startup indicator = 0
                else:
                    return m.p_total[g, t] == (m.P_MIN[g] * m.u[g, t]) + m.p[g, t]

            # Slow-start thermal generators
            elif g in m.G_THERM_SLOW:
                # Startup duration
                SU_D = ceil(m.P_MIN[g] / m.RR_SU[g])

                # Startup power output trajectory increment
                ramp_up_increment = m.P_MIN[g] / SU_D

                # Startup power output trajectory
                P_SU = OrderedDict({k + 1: ramp_up_increment * k for k in range(0, SU_D + 1)})

                # Shutdown duration
                SD_D = ceil(m.P_MIN[g] / m.RR_SD[g])

                # Shutdown power output trajectory increment
                ramp_down_increment = m.P_MIN[g] / SD_D

                # Shutdown power output trajectory
                P_SD = OrderedDict({k + 1: m.P_MIN[g] - (ramp_down_increment * k) for k in range(0, SD_D + 1)})

                if t != m.T.last():
                    return (m.p_total[g, t]
                            == ((m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
                                + sum(P_SU[k] * m.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                      range(1, SU_D + 1))
                                + sum(P_SD[k] * m.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                      range(2, SD_D + 2))))
                else:
                    return (m.p_total[g, t]
                            == ((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
                                + sum(P_SU[k] * m.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                      range(1, SU_D + 1))
                                + sum(P_SD[k] * m.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                      range(2, SD_D + 2))))
            else:
                raise Exception(f'Unexpected generator: {g}')

        # Constraint yielding total power output
        m.TOTAL_POWER_THERMAL = Constraint(m.G_THERM, m.T, rule=total_power_thermal_rule)

        def max_power_output_thermal_rule(_m, g, t):
            """Ensure max power + reserve is always less than installed capacity for thermal generators"""

            return m.p_total[g, t] + m.r_up[g, t] <= m.P_MAX[g]

        # Max power output + reserve is always less than installed capacity
        m.MAX_POWER_THERMAL = Constraint(m.G_THERM, m.T, rule=max_power_output_thermal_rule)

        def max_power_output_wind_rule(_m, g, t):
            """Max power output from wind generators"""

            return m.p_total[g, t] <= m.P_WIND[g, t]

        # Max power output from wind generators
        m.MAX_POWER_WIND = Constraint(m.G_WIND, m.T, rule=max_power_output_wind_rule)

        def max_power_output_solar_rule(_m, g, t):
            """Max power output from solar generators"""

            return m.p_total[g, t] <= m.P_SOLAR[g, t]

        # Max power output from wind generators
        m.MAX_POWER_SOLAR = Constraint(m.G_SOLAR, m.T, rule=max_power_output_solar_rule)

        def max_power_output_hydro_rule(_m, g, t):
            """Max power output from hydro generators"""

            return m.p_total[g, t] <= m.P_HYDRO[g, t]

        # Max power output from hydro generators
        m.MAX_POWER_HYDRO = Constraint(m.G_HYDRO, m.T, rule=max_power_output_hydro_rule)

        def storage_max_power_out_rule(_m, g, t):
            """
            Maximum discharging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely discharge in 1 hour
            """

            return m.p_in[g, t] <= m.P_MAX[g]

        # Max MW out of storage device - discharging
        m.P_STORAGE_MAX_OUT = Constraint(m.G_STORAGE, m.T, rule=storage_max_power_out_rule)

        def storage_max_power_in_rule(_m, g, t):
            """
            Maximum charging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely charge in 1 hour
            """

            return m.p_out[g, t] + m.r_up[g, t] <= m.P_MAX[g]

        # Max MW into storage device - charging
        m.P_STORAGE_MAX_IN = Constraint(m.G_STORAGE, m.T, rule=storage_max_power_in_rule)

        def max_storage_energy_rule(_m, g, t):
            """Ensure storage unit energy is within unit's capacity"""

            return m.q[g, t] <= m.Q_MAX[g]

        # Storage unit energy is within unit's limits
        m.STORAGE_ENERGY_BOUNDS = Constraint(m.G_STORAGE, m.T, rule=max_storage_energy_rule)

        def storage_energy_transition_rule(_m, g, t):
            """Constraint that couples energy + power between periods for storage units"""

            # If not the first period
            if t != m.T.first():
                return (m.q[g, t]
                        == m.q[g, t - 1] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t])
                        - (m.p_out[g, t] / m.BATTERY_EFFICIENCY[g]))
            else:
                # Assume battery completely discharged in first period (given by m.Q0)
                return (m.q[g, t]
                        == m.Q0[g] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t])
                        - (m.p_out[g, t] / m.BATTERY_EFFICIENCY[g]))

        # Account for inter-temporal energy transition within storage units
        m.STORAGE_ENERGY_TRANSITION = Constraint(m.G_STORAGE, m.T, rule=storage_energy_transition_rule)

        def storage_interval_end_lower_bound_rule(_m, g):
            """Ensure energy within storage unit at end of interval is greater than desired lower bound"""

            return m.Q_INTERVAL_END_LB[g] <= m.q[g, m.T.last()]

        # Ensure energy in storage unit at end of interval is above some desired lower bound
        m.STORAGE_INTERVAL_END_LOWER_BOUND = Constraint(m.G_STORAGE, rule=storage_interval_end_lower_bound_rule)

        def storage_interval_end_upper_bound_rule(_m, g):
            """
            Ensure energy within storage unit at end of interval is less than desired upper bound

            Note: Assuming upper bound for desired energy in unit at end of interval = installed capacity
            """

            return m.q[g, m.T.last()] <= m.Q_MAX[g]

        # Ensure energy in storage unit at end of interval is above some desired lower bound
        m.STORAGE_INTERVAL_END_UPPER_BOUND = Constraint(m.G_STORAGE, rule=storage_interval_end_upper_bound_rule)

        def power_balance_rule(_m, z, t):
            """Power balance for each NEM zone"""

            # Existing units within zone
            generators = [gen for gen, zone in self.data.generators.loc[:, 'NEM_ZONE'].items() if zone == z]

            # Storage units within a given zone
            storage_units = [gen for gen, zone in self.data.storage.loc[:, 'NEM_ZONE'].items() if zone == z]

            return (sum(m.p_total[g, t] for g in generators)
                    - m.DEMAND[z, t]
                    - sum(m.INCIDENCE_MATRIX[l, z] * m.p_flow[l, t] for l in m.L)
                    + sum(m.p_out[g, t] - m.p_in[g, t] for g in storage_units)
                    + m.p_V[z, t]
                    == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

        def powerflow_lower_bound_rule(_m, l, t):
            """Minimum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] >= m.POWERFLOW_MIN[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MIN_CONS = Constraint(m.L_I, m.T, rule=powerflow_lower_bound_rule)

        def powerflow_max_constraint_rule(_m, l, t):
            """Maximum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] <= m.POWERFLOW_MAX[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MAX_CONS = Constraint(m.L_I, m.T, rule=powerflow_max_constraint_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Define unit commitment problem objective function"""

        # Objective function
        m.OBJECTIVE = Objective(expr=m.OBJECTIVE_FUNCTION, sense=minimize)

        return m

    def construct_model(self, overlap):
        """Construct unit commitment model"""

        # Initialise model object
        m = ConcreteModel()

        # Add component allowing dual variables to be imported
        m.dual = Suffix(direction=Suffix.IMPORT)

        # Define sets
        m = self.define_sets(m, overlap)

        # Define parameters common to both UC and MPC models
        m = self.common.define_parameters(m)

        # Define parameters specific to unit commitment sub-problem
        m = self.define_parameters(m)

        # Define variables
        m = self.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m

    def update_demand(self, m, year, week, day):
        """Update demand for a given day"""

        # New demand values
        new_demand = {k: v for k, v in self.data.demand[(year, week, day)].items() if k[1] in m.T}

        # Update demand value in model object
        m.DEMAND.store_values(new_demand)

        return m

    def update_wind(self, m, year, week, day):
        """Update max power output for wind generators"""

        # Initialise container for updated wind data values
        new_values = {}

        # New output values
        for k, v in self.data.dispatch[(year, week, day)].items():

            # Check that element refers to generator in model, and time index within interval
            if (k[0] in m.G_WIND) and (k[1] in m.T):

                # Ensure wind data values are greater than zero and not too small (prevent numerical instability)
                if v < 0.1:
                    v = float(0)

                # Update container
                new_values[k] = float(v)

        # Update wind value in model object
        m.P_WIND.store_values(new_values)

        return m

    def update_solar(self, m, year, week, day):
        """Update max power output for solar generators"""

        # Initialise container for updated solar data values
        new_values = {}

        # New output values
        for k, v in self.data.dispatch[(year, week, day)].items():

            # Check that element refers to generator in model, and time index within interval
            if (k[0] in m.G_SOLAR) and (k[1] in m.T):

                # Ensure wind data values are greater than zero and not too small (prevent numerical instability)
                if v < 0.1:
                    v = float(0)

                # Update container
                new_values[k] = float(v)

        # Update solar values in model object
        m.P_SOLAR.store_values(new_values)

        return m

    def update_hydro(self, m, year, week, day):
        """Update max power output for hydro generators"""

        # Initialise container for updated hydro data values
        new_values = {}

        # New output values
        for k, v in self.data.dispatch[(year, week, day)].items():

            # Check that element refers to generator in model, and time index within interval
            if (k[0] in m.G_HYDRO) and (k[1] in m.T):

                # Ensure wind data values are greater than zero and not too small (prevent numerical instability)
                if v < 0.1:
                    v = float(0)

                # Update container
                new_values[k] = float(v)

        # Update hydro values in model object
        m.P_HYDRO.store_values(new_values)

        return m

    def update_parameters(self, m, year, month, day):
        """Update model parameters for a given day"""

        # Update demand
        m = self.update_demand(m, year, month, day)

        # Update wind power output
        m = self.update_wind(m, year, month, day)

        # Update solar output
        m = self.update_solar(m, year, month, day)

        # Update hydro output
        m = self.update_hydro(m, year, month, day)

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status

    @staticmethod
    def fix_binary_variables(m):
        """Fix all binary variables"""

        for k in m.u.keys(): m.u[k].fix(round(m.u[k].value))
        for k in m.v.keys(): m.v[k].fix(round(m.v[k].value))
        for k in m.w.keys(): m.w[k].fix(round(m.w[k].value))

        return m

    @staticmethod
    def unfix_binary_variables(m):
        """Fix all binary variables"""

        m.u.unfix()
        m.v.unfix()
        m.w.unfix()

        return m

    @staticmethod
    def save_solution(m, year, week, day, output_dir, update=False):
        """Save solution"""

        # Primal objects to extract
        primal = ['u', 'v', 'w', 'p_in', 'p_out', 'q', 'p', 'p_total', 'p_flow', 'p_V', 'r_up']

        # Dual objects to extract
        dual = ['POWER_BALANCE']

        # Expressions
        expressions = {'DAY_EMISSIONS': m.DAY_EMISSIONS.expr(),
                       'DAY_DEMAND': m.DAY_EMISSIONS.expr(),
                       'DAY_EMISSIONS_INTENSITY': m.DAY_EMISSIONS_INTENSITY.expr(),
                       'DAY_SCHEME_REVENUE': m.DAY_SCHEME_REVENUE.expr(),
                       'e': {k: m.e[k].expr() for k in m.e.keys()}}

        # Primal results
        primal_results = {v: m.__getattribute__(v).get_values() for v in primal}

        # Dual results
        dual_results = {d: {k: m.dual[v] for k, v in m.__getattribute__(d).items()} for d in dual}

        if update:
            # Load previously saved results
            with open(os.path.join(output_dir, f'interval_{year}_{week}_{day}.pickle'), 'rb') as f:
                previous_results = pickle.load(f)

            # Use previous power balance dual values for first 24 hours
            dual_results = {d: {k: previous_results[d][k] if k[1] <= 24 else dual_results[d][k]
                            for k in dual_results[d].keys()} for d in dual}

        # Combine primal a single dictionary
        results = {**primal_results, **dual_results, **expressions}

        # Save results
        with open(os.path.join(output_dir, f'interval_{year}_{week}_{day}.pickle'), 'wb') as f:
            pickle.dump(results, f)

        return results

    @staticmethod
    def fix_interval_overlap(m, year, week, day, overlap, output_dir):
        """Fix variables at the start of a given dispatch interval based on the solution of the preceding interval"""

        if day == 1:
            week = week - 1
            day = 7
        else:
            day = day - 1

        # Load solution for previous day
        previous_solution = pd.read_pickle(os.path.join(output_dir, f'interval_{year}_{week}_{day}.pickle'))

        # Map time index between beginning of current day and end of preceding interval
        interval_map = {i: 24 + i for i in range(0, overlap + 1)}

        # Fix variables to values obtained in preceding interval
        for t in range(1, overlap + 1):
            for g in m.G.difference(m.G_STORAGE, m.G_THERM):
                m.p_total[g, t].fix(previous_solution['p_total'][(g, interval_map[t])])

            for g in m.G_THERM:
                m.u[g, t].fix(round(previous_solution['u'][(g, interval_map[t])]))
                m.v[g, t].fix(round(previous_solution['v'][(g, interval_map[t])]))
                m.w[g, t].fix(round(previous_solution['w'][(g, interval_map[t])]))
                m.p[g, t].fix(previous_solution['p'][(g, interval_map[t])])
                m.r_up[g, t].fix(previous_solution['r_up'][(g, interval_map[t])])

                m.U0[g] = int(round(previous_solution['u'][(g, interval_map[0])]))
                m.P0[g] = previous_solution['p'][(g, interval_map[0])]

            for g in m.G_STORAGE:
                m.q[g, t].fix(previous_solution['q'][(g, interval_map[t])])
                m.p_in[g, t].fix(previous_solution['p_in'][(g, interval_map[t])])
                m.p_out[g, t].fix(previous_solution['p_out'][(g, interval_map[t])])
                m.r_up[g, t].fix(previous_solution['r_up'][(g, interval_map[t])])

                m.Q0[g] = previous_solution['q'][(g, interval_map[0])]

            for l in m.L:
                m.p_flow[l, t].fix(previous_solution['p_flow'][(l, interval_map[t])])

            for z in m.Z:
                m.p_V[z, t].fix(previous_solution['p_V'][(z, interval_map[t])])

        return m

    @staticmethod
    def fix_interval(m, start, end):
        """Fix all variables for a defined interval"""

        # Fix variables to values obtained in preceding interval
        for t in range(start, end + 1):
            for g in m.G.difference(m.G_STORAGE, m.G_THERM):
                m.p_total[g, t].fix()

            for g in m.G_THERM:
                m.u[g, t].fix(round(m.u[g, t].value))
                m.v[g, t].fix(round(m.v[g, t].value))
                m.w[g, t].fix(round(m.w[g, t].value))
                m.p[g, t].fix()
                m.r_up[g, t].fix()

            for g in m.G_STORAGE:
                m.q[g, t].fix()
                m.p_in[g, t].fix()
                m.p_out[g, t].fix()
                m.r_up[g, t].fix()

            for l in m.L:
                m.p_flow[l, t].fix()

            for z in m.Z:
                m.p_V[z, t].fix()

        return m

    @staticmethod
    def unfix_interval(m, start, end):
        """Fix all variables for a defined interval"""

        # Fix variables to values obtained in preceding interval
        for t in range(start, end + 1):
            for g in m.G.difference(m.G_STORAGE, m.G_THERM):
                m.p_total[g, t].unfix()

            for g in m.G_THERM:
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()
                m.p[g, t].unfix()
                m.r_up[g, t].unfix()

            for g in m.G_STORAGE:
                m.q[g, t].unfix()
                m.p_in[g, t].unfix()
                m.p_out[g, t].unfix()
                m.r_up[g, t].unfix()

            for l in m.L:
                m.p_flow[l, t].unfix()

            for z in m.Z:
                m.p_V[z, t].unfix()

        return m


def cleanup_pickle(directory):
    """Delete pickle files in a directory"""

    # Pickle files
    files = [f for f in os.listdir(directory) if ('.pickle' in f) and (('interval' in f) or ('mpc' in f))]

    for f in files:
        os.remove(os.path.join(directory, f))


if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')
