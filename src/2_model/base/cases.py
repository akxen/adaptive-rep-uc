"""Cases run using MPC updating algorithm"""

import os
import pickle

import numpy as np

from uc import UnitCommitment
from mpc import MPCController
from data import ModelData
from forecast import PersistenceForecast, MonteCarloForecast
from analysis import AnalyseResults


class ModelCases:
    def __init__(self):
        self.data = ModelData()

    @staticmethod
    def cleanup_directory(directory):
        """Delete selected result files in a directory"""

        # Pickle files
        files = [f for f in os.listdir(directory) if ('.pickle' in f) and (('interval' in f) or ('mpc' in f))]

        for f in files:
            os.remove(os.path.join(directory, f))

    @staticmethod
    def create_case_directory(case_output_dir):
        """Create directory for case output files if it does not already exist"""

        # Create directory if it doesn't already exist
        if not os.path.exists(case_output_dir):
            os.mkdir(case_output_dir)

        return case_output_dir

    @staticmethod
    def run_solve_sequence(cls, m):
        """Run UC solve sequence

        Parameters
        ----------
        cls : class
            Class used to construct UC model object. Contains method to solve UC model instance.

        m : pyomo model
            Unit commitment model object.

        Returns
        -------
        m : pyomo model
            Unit commitment model object after applying solver.

        flag : bool
            Break flag. If 'True' model is infeasible.
        """

        # Solve model
        m, status_mip = cls.solve_model(m)

        if status_mip['Solver'][0]['Termination condition'].key != 'optimal':
            flag = True
            return m, flag

        # Fix binary variables
        m = cls.fix_binary_variables(m)

        # Re-solve to obtain prices
        m, status_lp = cls.solve_model(m)

        if status_lp['Solver'][0]['Termination condition'].key != 'optimal':
            flag = True
            return m, flag

        # Break flag
        flag = False

        return m, flag

    def run_case(self, params, hot_start=None):
        """Run case parameters"""

        # Save case parameters
        with open(os.path.join(params['output_dir'], 'parameters.pickle'), 'wb') as f:
            pickle.dump(params, f)

        # Unit commitment and MPC model objects
        uc = UnitCommitment()
        mpc = MPCController()

        # Objects used to generate forecasts for MPC updating model and analyse model results
        persistence_forecast = PersistenceForecast()
        scenario_forecast = MonteCarloForecast()
        analysis = AnalyseResults()

        # Construct UC and MPC models
        m_uc = uc.construct_model(params['overlap_intervals'])
        m_mpc = mpc.construct_model(generators=m_uc.G, n_intervals=params['calibration_intervals'],
                                    n_scenarios=params['scenarios'])

        # Activate additional constraints corresponding to different cases
        if params['case_name'] == 'revenue_floor':
            m_mpc.REVENUE_FLOOR_CONS.activate()

        # Initialise policy parameters (baseline and permit price)
        m_uc.PERMIT_PRICE.store_values(params['permit_price'])

        for t in m_uc.T:
            if params['case_name'] == 'carbon_tax':
                m_uc.BASELINE[t] = float(0)
            else:
                m_uc.BASELINE[t] = float(params['baseline_start'])

        # Counter for model windows, and flag used to break loop if model is infeasible
        window = 1
        break_flag = False

        # Years and weeks over which to iterate. Adjust if having a hot-start
        if hot_start is not None:
            years = range(hot_start[0], max(params['years']) + 1)
            weeks = range(hot_start[1], 53)
        else:
            years = params['years']
            weeks = range(1, 53)

        for y in years:
            if break_flag:
                break

            for w in weeks:
                if break_flag:
                    break

                for d in params['days']:
                    print(f'Running window {window}: year={y}, week={w}, day={d}')

                    # Update model parameters for a given day
                    m_uc = uc.update_parameters(m_uc, y, w, d)

                    if window != 1:
                        # Fix interval start using solution from previous window
                        m_uc = uc.fix_interval_overlap(m_uc, y, w, d, params['overlap_intervals'], params['output_dir'])

                    # Run solve sequence. First solve MILP, then fix integer variables and re-solve to obtain prices.
                    m_uc, break_flag = self.run_solve_sequence(uc, m_uc)

                    # Break loop if model is infeasible
                    if break_flag:
                        break

                    # Save solution
                    uc.save_solution(m_uc, y, w, d, params['output_dir'])

                    # Unfix binary variables
                    m_uc = uc.unfix_binary_variables(m_uc)

                    # Check if next week will be the last calibration interval in the model horizon
                    next_week_is_last_interval = (w == max(params['weeks'])) and (y == max(params['years']))

                    # If not the last week, and the baseline can be updated
                    if (d == 7) and (not next_week_is_last_interval) and params['baseline_update_required']:

                        # Year and week index for next interval. Take into account year changing
                        if w == max(params['weeks']):
                            next_w = 1
                            next_y = y + 1
                        else:
                            next_w = w + 1
                            next_y = y

                        # Get cumulative scheme revenue
                        cumulative_revenue = analysis.get_cumulative_scheme_revenue(params['output_dir'], next_y,
                                                                                    next_y)

                        # Get generator energy forecast for next set of calibration intervals
                        if (params['case_name'] == 'multi_scenario_forecast') and (next_y == 2018):
                            energy_forecast, probabilities = scenario_forecast.get_scenarios(
                                output_dir=params['output_dir'],
                                year=next_y,
                                week=next_w,
                                start_year=min(params['years']),
                                n_intervals=params['calibration_intervals'],
                                n_random_paths=params['n_random_paths'],
                                n_clusters=params['scenarios'],
                                eligible_generators=m_mpc.G)

                        else:
                            # Use a persistence-based forecast
                            energy_forecast, probabilities = persistence_forecast.get_energy_forecast_persistence(
                                output_dir=params['output_dir'],
                                year=next_y,
                                week=next_w,
                                n_intervals=params['calibration_intervals'],
                                eligible_generators=m_mpc.G)

                        # Update emissions intensities if there is an anticipated emissions intensity shock
                        if ((params['case_name'] == 'anticipated_emissions_intensity_shock')
                                and (next_y == params['emissions_shock_year'])
                                and (next_w > params['emissions_shock_week'] - params['calibration_intervals'])):

                            # Emissions intensities for future calibration intervals when shock is anticipated
                            for g in m_mpc.G:
                                for c in range(max(params['emissions_shock_week'] - next_w + 1, 1),
                                               params['calibration_intervals'] + 1):
                                    # Update emissions intensities if shock week within the forecast horizon
                                    m_mpc.EMISSIONS_RATE[g, c] = (params['emissions_shock_factor'][g]
                                                                  * float(self.data.generators.loc[g, 'EMISSIONS']))

                        # Compute revenue target to use when updating baselines
                        revenue_target = self.get_mpc_revenue_target_input(next_y, next_w, params['revenue_target'],
                                                                           params['calibration_intervals'])

                        # Get updated baselines
                        mpc_results = mpc.run_baseline_updater(m_mpc, next_y, next_w,
                                                               baseline_start=m_uc.BASELINE[1].value,
                                                               revenue_start=cumulative_revenue,
                                                               revenue_target=revenue_target,
                                                               revenue_floor=params['revenue_floor'],
                                                               permit_price={k: v.value for k, v in
                                                                             m_uc.PERMIT_PRICE.items()},
                                                               energy_forecast=energy_forecast,
                                                               scenario_probabilities=probabilities)

                        # Save MPC results
                        mpc.save_results(next_y, next_w, mpc_results, params['output_dir'])

                        # Update baseline (starting at beginning of following day)
                        for h in [t for t in m_uc.T if t > 24]:
                            m_uc.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                        # Fix variables up until end of day (beginning of overlap period for next day)
                        m_uc = uc.fix_interval(m_uc, start=1, end=24)

                        # Run solve sequence. First solve MILP, then fix integer variables and re-solve to obtain prices
                        m_uc, break_flag = self.run_solve_sequence(uc, m_uc)

                        # Break loop if model is infeasible
                        if break_flag:
                            break

                        # Save solution (updates previously saved solution for this interval)
                        uc.save_solution(m_uc, y, w, d, params['output_dir'], update=True)

                        # Unfix binary variables
                        m_uc = uc.unfix_binary_variables(m_uc)

                        # Unfix remaining variables
                        m_uc = uc.unfix_interval(m_uc, start=1, end=24)

                        # All intervals = baseline obtained from MPC model in preparation for next iteration.
                        for h in m_uc.T:
                            m_uc.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                    # Apply unanticipated emissions intensity shock if required
                    if ((params['case_name'] == 'unanticipated_emissions_intensity_shock')
                            and (y == params['emissions_shock_year']) and (w == params['emissions_shock_week'])
                            and (d == 1)):
                        # Applying new emissions intensities for coming calibration interval (misaligned with forecast)
                        for g in m_uc.G:
                            print(f'UC old emissions intensity {g}: {m_uc.EMISSIONS_RATE[g].value}')
                            m_uc.EMISSIONS_RATE[g] = params['emissions_shock_factor'][g] * m_uc.EMISSIONS_RATE[g].value
                            print(f'UC new emissions intensity {g}: {m_uc.EMISSIONS_RATE[g].value}')

                        # Emissions intensities aligned for next calibration interval
                        for g in m_mpc.G:
                            for c in m_mpc.C:
                                print(f'MPC old emissions intensity {g}: {m_mpc.EMISSIONS_RATE[g, c].value}')
                                m_mpc.EMISSIONS_RATE[g, c] = (params['emissions_shock_factor'][g]
                                                              * m_mpc.EMISSIONS_RATE[g, c].value)
                                print(f'MPC new emissions intensity {g}: {m_mpc.EMISSIONS_RATE[g, c].value}')

                    # Update rolling window counter
                    window += 1

    @staticmethod
    def get_revenue_target(years, weeks):
        """Construct revenue target to raise 30,000,000 over 10 calibration intervals"""

        # Positive revenue target. Earn 10,000,000 over 10 calibration intervals
        revenue_target = {}
        for i, year in enumerate(years):
            revenue_target[year] = {}

            for week in weeks:

                # If in the first year, before the revenue ramp-up period
                if (week < 10) and (i == 0):
                    revenue_target[year][week] = 0

                # If in the revenue ramp-up period
                elif (week >= 10) and (week <= 20) and (i == 0):
                    revenue_target[year][week] = (week - 10) * 3000000

                # Else, keep the revenue target constant
                else:
                    revenue_target[year][week] = 30 * 1000000

            return revenue_target

    @staticmethod
    def get_mpc_revenue_target_input(year, week, revenue_trajectory, calibration_intervals):
        """
        Given a revenue target trajectory, determine the revenue target used in the MPC updating protocol. Note
        that the MPC program looks forward in time, and the target will become relevant before the first ramp-up week.
        """

        # If at the end of given year, update the
        if week + calibration_intervals > 52:
            target_year = year + 1
            target_week = week + calibration_intervals - 52

        else:
            target_year = year
            target_week = week + calibration_intervals

        # Revenue target at the end of the calibration interval horizon. Use value in final year of trajectory
        # if out of bounds.
        try:
            revenue_target = revenue_trajectory[target_year][target_week]
        except Exception as e:
            print(e)
            print('Using revenue target in last week of revenue trajectory')
            final_year = max(revenue_trajectory.keys())
            final_week = max(revenue_trajectory[final_year].keys())
            revenue_target = revenue_trajectory[final_year][final_week]

        return revenue_target

    def generate_cases(self, years, weeks, output_dir):
        """Generate cases to run"""

        # UC dummy object
        uc_d = UnitCommitment()
        m_d = uc_d.construct_model(overlap=1)

        # Days to be used in each case (1-7)
        days = range(1, 8)

        # Positive revenue target
        revenue_target = self.get_revenue_target(years, weeks)

        # Emissions intensity shock at week 10
        np.random.seed(10)
        emissions_shock_factor = {g: float(np.random.uniform(0.7, 1)) if g in m_d.G_THERM else float(1) for g in m_d.G}

        # BAU case
        case_params = {'bau':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 1,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'bau',
                            'output_dir': os.path.join(output_dir, 'bau'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(0) for g in m_d.G},
                            'baseline_update_required': False},

                       'carbon_tax':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 1,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'carbon_tax',
                            'output_dir': os.path.join(output_dir, 'carbon_tax'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) for g in m_d.G},
                            'baseline_update_required': False},

                       'revenue_target_1_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 1,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'revenue_target',
                            'output_dir': os.path.join(output_dir, 'revenue_target'),
                            'revenue_floor': None,
                            'revenue_target': revenue_target,
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'baseline_update_required': True},

                       'revenue_target_3_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 3,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'revenue_target',
                            'output_dir': os.path.join(output_dir, 'revenue_target'),
                            'revenue_floor': None,
                            'revenue_target': revenue_target,
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'baseline_update_required': True},

                       'anticipated_emissions_intensity_shock_1_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 1,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'anticipated_emissions_intensity_shock',
                            'output_dir': os.path.join(output_dir, 'anticipated_emissions_intensity_shock'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'emissions_shock_factor': emissions_shock_factor,
                            'emissions_shock_year': 2018,
                            'emissions_shock_week': 10,
                            'baseline_update_required': True},

                       'anticipated_emissions_intensity_shock_3_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 3,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'anticipated_emissions_intensity_shock',
                            'output_dir': os.path.join(output_dir, 'anticipated_emissions_intensity_shock'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'emissions_shock_factor': emissions_shock_factor,
                            'emissions_shock_year': 2018,
                            'emissions_shock_week': 10,
                            'baseline_update_required': True},

                       'unanticipated_emissions_intensity_shock_1_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 1,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'unanticipated_emissions_intensity_shock',
                            'output_dir': os.path.join(output_dir, 'unanticipated_emissions_intensity_shock'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'emissions_shock_factor': emissions_shock_factor,
                            'emissions_shock_year': 2018,
                            'emissions_shock_week': 10,
                            'baseline_update_required': True},

                       'unanticipated_emissions_intensity_shock_3_ci':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 3,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'unanticipated_emissions_intensity_shock',
                            'output_dir': os.path.join(output_dir, 'unanticipated_emissions_intensity_shock'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'emissions_shock_factor': emissions_shock_factor,
                            'emissions_shock_year': 2018,
                            'emissions_shock_week': 10,
                            'baseline_update_required': True},

                       'renewables_eligibility':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 3,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'renewables_eligibility',
                            'output_dir': os.path.join(output_dir, 'renewables_eligibility'),
                            'revenue_floor': None,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM.union(m_d.G_WIND, m_d.G_SOLAR) else
                            float(0) for g in m_d.G},
                            'baseline_update_required': True},

                       'revenue_floor':
                           {'years': years, 'weeks': weeks, 'days': days,
                            'overlap_intervals': 17,
                            'calibration_intervals': 3,
                            'scenarios': 1,
                            'baseline_start': 1,
                            'case_name': 'revenue_floor',
                            'output_dir': os.path.join(output_dir, 'revenue_floor'),
                            'revenue_floor': -1e6,
                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in m_d.G},
                            'baseline_update_required': True},
                       }

        # Testing various calibration interval durations
        calibration_interval_params = {f'{i}_calibration_intervals':
                                           {'years': years, 'weeks': weeks, 'days': days,
                                            'overlap_intervals': 17,
                                            'calibration_intervals': i,
                                            'scenarios': 1,
                                            'baseline_start': 1,
                                            'case_name': f'{i}_calibration_intervals',
                                            'output_dir': os.path.join(output_dir, f'{i}_calibration_intervals'),
                                            'revenue_floor': None,
                                            'revenue_target': {y: {w: float(0) for w in weeks} for y in years},
                                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in
                                                             m_d.G},
                                            'baseline_update_required': True}
                                       for i in range(1, 7)}

        # Persistence-based forecast - 3 calibration intervals for 2017-2018
        persistence_forecast_params = {f'persistence_forecast':
                                           {'years': [2017, 2018], 'weeks': weeks, 'days': days,
                                            'overlap_intervals': 17,
                                            'calibration_intervals': 3,
                                            'scenarios': 1,
                                            'baseline_start': 1,
                                            'case_name': f'persistence_forecast',
                                            'output_dir': os.path.join(output_dir, f'persistence_forecast'),
                                            'revenue_floor': None,
                                            'revenue_target': {y: {w: float(0) for w in weeks} for y in [2017, 2018]},
                                            'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in
                                                             m_d.G},
                                            'baseline_update_required': True},
                                       }

        # Testing probabilistic forecast method
        multi_scenario_forecast_params = {'multi_scenario_forecast':
                                              {'years': [2017, 2018], 'weeks': weeks, 'days': days,
                                               'overlap_intervals': 17,
                                               'calibration_intervals': 3,
                                               'scenarios': 5,
                                               'n_random_paths': 500,
                                               'baseline_start': 1,
                                               'case_name': 'multi_scenario_forecast',
                                               'output_dir': os.path.join(output_dir, 'multi_scenario_forecast'),
                                               'revenue_floor': None,
                                               'revenue_target': {y: {w: float(0) for w in weeks} for y in
                                                                  [2017, 2018]},
                                               'permit_price': {g: float(40) if g in m_d.G_THERM else float(0) for g in
                                                                m_d.G},
                                               'baseline_update_required': True
                                               }
                                          }

        # Combine all cases into a single dictionary
        case_params = {**case_params, **calibration_interval_params, **persistence_forecast_params,
                       **multi_scenario_forecast_params}

        return case_params


if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Object used to construct and run model cases
    cases = ModelCases()

    # Generate case parameters for given year and week ranges
    case_parameters = cases.generate_cases(years=[2018], weeks=range(1, 53), output_dir=output_directory)

    # Run all cases
    for name, parameters in case_parameters.items():
        print(f'Running case: {name}')
        print(parameters)

        # Create directory if it doesn't exist, and cleanup files
        # cases.create_case_directory(parameters['output_dir'])
        # cases.cleanup_directory(parameters['output_dir'])

        # Run case
        cases.run_case(parameters, hot_start=(2018, 1))
