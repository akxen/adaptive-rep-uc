"""Model cases to run"""

import os

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


def run_solve_sequence(m):
    """Run UC solve sequence

    Parameters
    ----------
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
    m, status_mip = uc.solve_model(m)

    if status_mip['Solver'][0]['Termination condition'].key != 'optimal':
        flag = True
        return m, flag

    # Fix binary variables
    m = uc.fix_binary_variables(m)

    # Re-solve to obtain prices
    m, status_lp = uc.solve_model(m)

    if status_lp['Solver'][0]['Termination condition'].key != 'optimal':
        flag = True
        return m, flag

    # Break flag
    flag = False

    return m, flag


if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Cleanup pickle files in output directory
    cleanup_pickle(output_directory)

    # Model parameters
    years = [2018]
    weeks = range(1, 53)
    days = range(1, 8)
    overlap_intervals = 17
    calibration_intervals = 1
    scenarios = 1
    baseline_start = 1

    # Unit commitment and MPC model objects
    uc = UnitCommitment()
    mpc = MPCController()

    # Objects used to generate forecasts for MPC updating model and analyse model results
    forecast = Forecast()
    analysis = AnalyseResults()

    # Construct UC and MPC models
    m_uc = uc.construct_model(overlap_intervals)
    m_mpc = mpc.construct_model(eligible_generators=m_uc.G_THERM, n_intervals=calibration_intervals,
                                n_scenarios=scenarios)

    # Initialise policy parameters (baseline and permit price)
    m_uc.PERMIT_PRICE = float(40)

    for t in m_uc.T:
        m_uc.BASELINE[t] = float(baseline_start)

    # Counter for model windows, and flag used to break loop if model is infeasible
    window = 1
    break_flag = False

    for y in years:
        if break_flag:
            break

        for w in weeks:
            if break_flag:
                break

            for d in days:
                print(f'Running window {window}: year={y}, week={w}, day={d}')

                # Update model parameters for a given day
                m_uc = uc.update_parameters(m_uc, y, w, d)

                if window != 1:
                    # Fix interval start using solution from previous window
                    m_uc = uc.fix_interval_overlap(m_uc, y, w, d, overlap_intervals, output_directory)

                # Run solve sequence. First solve MILP, then fix integer variables and re-solve to obtain prices.
                m_uc, break_flag = run_solve_sequence(m_uc)

                # Break loop if model is infeasible
                if break_flag:
                    break

                # Save solution
                solution = uc.save_solution(m_uc, y, w, d, output_directory)

                # Unfix binary variables
                m_uc = uc.unfix_binary_variables(m_uc)

                if (d == 7) and (w <= 51):
                    # Get cumulative scheme revenue
                    cumulative_revenue = analysis.get_cumulative_scheme_revenue(output_directory, y, w + 1)

                    # Get generator energy forecast for following calibration intervals
                    energy_forecast, probabilities = forecast.get_energy_forecast_persistence(
                        output_dir=output_directory,
                        year=y,
                        week=w + 1,
                        n_intervals=calibration_intervals,
                        eligible_generators=m_mpc.G)

                    # Get updated baselines
                    mpc_results = mpc.run_baseline_updater(m_mpc, y, w + 1,
                                                           baseline_start=m_uc.BASELINE[1].value,
                                                           revenue_start=cumulative_revenue,
                                                           revenue_target=0,
                                                           revenue_floor=float(-1e6),
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
                    m_uc, break_flag = run_solve_sequence(m_uc)

                    # Break loop if model is infeasible
                    if break_flag:
                        break

                    # Save solution (updates previously saved solution for this interval)
                    solution = uc.save_solution(m_uc, y, w, d, output_directory)

                    # Unfix binary variables
                    m_uc = uc.unfix_binary_variables(m_uc)

                    # Unfix remaining variables
                    m_uc = uc.unfix_interval(m_uc, start=1, end=24)

                    # All intervals = baseline obtained from MPC model in preparation for next iteration.
                    for h in m_uc.T:
                        m_uc.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                # Update rolling window counter
                window += 1
