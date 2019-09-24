"""Model cases to run"""

import os

import numpy as np

from uc import UnitCommitment
from mpc import MPCController
from forecast import Forecast
from analysis import AnalyseResults


def cleanup_pickle(directory):
    """Delete pickle files in a directory"""

    # Pickle files
    files = [f for f in os.listdir(directory) if ('.pickle' in f) and (('interval' in f) or ('mpc' in f))]

    for f in files:
        os.remove(os.path.join(directory, f))


def create_case_output_directory(case_output_dir):
    """Create directory for case output files if it does not already exist"""

    # Create directory if it doesn't already exist
    if not os.path.exists(case_output_dir):
        os.mkdir(case_output_dir)

    return case_output_dir


def run_solve_sequence(m_obj, m):
    """Run UC solve sequence

    Parameters
    ----------
    m_obj : class
        Class used to construct UC model object

    m : pyomo model
        Unit commitment model object

    Returns
    -------
    m : pyomo model
        Unit commitment model object after applying solver

    flag : bool
        Break flag. If 'True' model is infeasible.
    """

    # Solve model
    m, status_mip = m_obj.solve_model(m)

    if status_mip['Solver'][0]['Termination condition'].key != 'optimal':
        flag = True
        return m, flag

    # Fix binary variables
    m = m_obj.fix_binary_variables(m)

    # Re-solve to obtain prices
    m, status_lp = m_obj.solve_model(m)

    if status_lp['Solver'][0]['Termination condition'].key != 'optimal':
        flag = True
        return m, flag

    # Break flag
    flag = False

    return m, flag


def run_case(params):
    """Run case parameters"""

    # Unit commitment and MPC model objects
    uc = UnitCommitment()
    mpc = MPCController()

    # Objects used to generate forecasts for MPC updating model and analyse model results
    forecast = Forecast()
    analysis = AnalyseResults()

    # Construct UC and MPC models
    m_uc = uc.construct_model(params['overlap_intervals'])
    m_mpc = mpc.construct_model(generators=m_uc.G, n_intervals=params['calibration_intervals'],
                                n_scenarios=params['scenarios'])

    # Initialise policy parameters (baseline and permit price)
    m_uc.PERMIT_PRICE = float(params['permit_price'])

    for t in m_uc.T:
        m_uc.BASELINE[t] = float(params['baseline_start'])

    # Counter for model windows, and flag used to break loop if model is infeasible
    window = 1
    break_flag = False

    for y in params['years']:
        if break_flag:
            break

        for w in params['weeks']:
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
                m_uc, break_flag = run_solve_sequence(uc, m_uc)

                # Break loop if model is infeasible
                if break_flag:
                    break

                # Save solution
                uc.save_solution(m_uc, y, w, d, output_directory)

                # Unfix binary variables
                m_uc = uc.unfix_binary_variables(m_uc)

                if (d == 7) and (w <= 51):
                    # Get cumulative scheme revenue
                    cumulative_revenue = analysis.get_cumulative_scheme_revenue(params['output_dir'], y, w + 1)

                    # Get generator energy forecast for following calibration intervals
                    energy_forecast, probabilities = forecast.get_energy_forecast_persistence(
                        output_dir=params['output_dir'],
                        year=y,
                        week=w + 1,
                        n_intervals=params['calibration_intervals'],
                        eligible_generators=m_mpc.G)

                    # Get updated baselines
                    mpc_results = mpc.run_baseline_updater(m_mpc, y, w + 1,
                                                           baseline_start=m_uc.BASELINE[1].value,
                                                           revenue_start=cumulative_revenue,
                                                           revenue_target=float(params['revenue_target'][w + 1]),
                                                           revenue_floor=float(params['revenue_floor']),
                                                           permit_price=m_uc.PERMIT_PRICE.value,
                                                           energy_forecast=energy_forecast,
                                                           scenario_probabilities=probabilities)

                    # Save MPC results
                    mpc.save_results(y, w + 1, mpc_results)

                    # Update baseline (starting at beginning of following day)
                    for h in [t for t in m_uc.T if t > 24]:
                        m_uc.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                    # Fix variables up until end of day (beginning of overlap period for next day)
                    m_uc = uc.fix_interval(m_uc, start=1, end=24)

                    # Run solve sequence. First solve MILP, then fix integer variables and re-solve to obtain prices.
                    m_uc, break_flag = run_solve_sequence(uc, m_uc)

                    # Break loop if model is infeasible
                    if break_flag:
                        break

                    # Save solution (updates previously saved solution for this interval)
                    uc.save_solution(m_uc, y, w, d, params['output_dir'])

                    # Unfix binary variables
                    m_uc = uc.unfix_binary_variables(m_uc)

                    # Unfix remaining variables
                    m_uc = uc.unfix_interval(m_uc, start=1, end=24)

                    # All intervals = baseline obtained from MPC model in preparation for next iteration.
                    for h in m_uc.T:
                        m_uc.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                # Update rolling window counter
                window += 1


if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # UC dummy object
    uc_dummy = UnitCommitment()
    m_uc_dummy = uc_dummy.construct_model(overlap=1)

    years, weeks, days = [2018], range(1, 53), range(1, 8)

    # BAU case
    benchmark_params = {'bau':
                            {'years': years, 'weeks': weeks, 'days': days,
                             'overlap_intervals': 17,
                             'calibration_intervals': 1,
                             'scenarios': 1,
                             'baseline_start': 1,
                             'case_name': 'bau',
                             'output_dir': os.path.join(output_directory, 'bau'),
                             'revenue_floor': -1e9,
                             'revenue_target': {w: 0 for w in range(1, 53)},
                             'permit_price': {g: float(0) for g in m_uc_dummy.G}},
                        'carbon_tax':
                            {'years': [2018], 'weeks': range(1, 53), 'days': range(1, 8),
                             'overlap_intervals': 17,
                             'calibration_intervals': 1,
                             'scenarios': 1,
                             'baseline_start': 1,
                             'case_name': 'carbon_tax',
                             'output_dir': os.path.join(output_directory, 'carbon_tax'),
                             'revenue_floor': -1e9,
                             'revenue_target': {w: 0 for w in range(1, 53)},
                             'permit_price': {g: float(0) for g in m_uc_dummy.G}}}

    # Testing various calibration interval durations
    ci_params = {f'{i}_calibration_intervals':
                     {'years': years, 'weeks': weeks, 'days': days,
                      'overlap_intervals': 17,
                      'calibration_intervals': i,
                      'scenarios': 1,
                      'baseline_start': 1,
                      'case_name': f'{i}_calibration_intervals',
                      'output_dir': os.path.join(output_directory, 'carbon_tax'),
                      'revenue_floor': -1e9,
                      'revenue_target': {w: 0 for w in range(1, 53)},
                      'permit_price': {g: float(40) if g in m_uc_dummy.G_THERM else float(0) for g in m_uc_dummy.G}}
                 for i in range(1, 7)}

    # Positive revenue target. Earn 10,000,000 over 10 calibration intervals. Using 4 lookahead intervals for updating.
    revenue_target = {}
    for w in range(1, 53):
        if w < 10:
            revenue_target[w] = 0
        elif (w >= 10) and (w <= 20):
            revenue_target[w] = (w - 10) * 100000
        else:
            revenue_target[w] = 10 * 1000000

    revenue_params = {'revenue_target':
                          {'years': years, 'weeks': weeks, 'days': days,
                           'overlap_intervals': 17,
                           'calibration_intervals': 4,
                           'scenarios': 1,
                           'baseline_start': 1,
                           'case_name': 'revenue_target',
                           'output_dir': os.path.join(output_directory, 'carbon_tax'),
                           'revenue_floor': -1e9,
                           'revenue_target': revenue_target,
                           'permit_price': {g: float(40) if g in m_uc_dummy.G_THERM else float(0) for g in m_uc_dummy.G}
                           }
                      }

    # Emissions intensity shock at week 10
    np.random.seed(10)
    emissions_shock_factor = {g: float(np.random.uniform(0.5, 1)) if g in m_uc_dummy.G_THERM else float(1)
                              for g in m_uc_dummy.G}
    shock_params = {'emissions_intensity_shock':
                        {'years': years, 'weeks': weeks, 'days': days,
                         'overlap_intervals': 17,
                         'calibration_intervals': 4,
                         'scenarios': 1,
                         'baseline_start': 1,
                         'case_name': 'emissions_intensity_shock',
                         'output_dir': os.path.join(output_directory, 'emissions_intensity_shock'),
                         'revenue_floor': -1e9,
                         'revenue_target': {w: 0 for w in range(1, 53)},
                         'permit_price': {g: float(40) if g in m_uc_dummy.G_THERM else float(0) for g in m_uc_dummy.G},
                         'emissions_shock_factor': emissions_shock_factor,
                         'emissions_shock_week': 10}
                    }

    # Altering eligibility criteria (solar and wind eligible to receive credits)
    eligibility_params = {'renewables_eligibility':
                              {'years': years, 'weeks': weeks, 'days': days,
                               'overlap_intervals': 17,
                               'calibration_intervals': 4,
                               'scenarios': 1,
                               'baseline_start': 1,
                               'case_name': 'renewables_eligibility',
                               'output_dir': os.path.join(output_directory, 'renewables_eligibility'),
                               'revenue_floor': -1e9,
                               'revenue_target': {w: 0 for w in range(1, 53)},
                               'permit_price': {g: float(40) if g in m_uc_dummy.G_THERM.union(m_uc_dummy.G_WIND, m_uc_dummy.G_SOLAR) else float(0) for g in m_uc_dummy.G}}
                          }

    # Combine all cases into a single dictionary
    case_params = {**benchmark_params, **ci_params, **revenue_params, **shock_params, **eligibility_params}

    for name, parameters in case_params.items():
        print(f'Running case: {name}')
        print(parameters)
        run_case(parameters)
