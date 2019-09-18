"""Model cases to run"""

import os

from uc import UnitCommitment
from mpc import MPCController


def cleanup_pickle(directory):
    """Delete pickle files in a directory"""

    # Pickle files
    files = [f for f in os.listdir(directory) if ('.pickle' in f) and (('interval' in f) or ('mpc' in f))]

    for f in files:
        os.remove(os.path.join(directory, f))


if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Cleanup pickle files in output directory
    cleanup_pickle(output_directory)

    # Model parameters
    years = [2018]
    weeks = range(1, 53)
    days = range(1, 8)
    interval_overlap = 17
    baseline_start = 1

    # Initialise object used to construct Unit Commitment model
    uc = UnitCommitment()

    # Initialise object used to construct MPC baseline updater
    mpc = MPCController()

    # MPC model
    mpc_model = mpc.construct_model()

    # Construct model object
    model = uc.construct_model()

    # Initialise permit price
    model.PERMIT_PRICE = float(40)

    # Initial emissions intensity baseline
    for t in model.T:
        model.BASELINE[t] = float(baseline_start)

    # Counter for model windows
    window = 1

    # r = pd.read_pickle(os.path.join(output_directory, f'interval_{2017}_{1}_{5}.pickle'))
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
                model = uc.update_parameters(model, y, w, d)

                if window != 1:
                    # Fix interval start using solution from previous window
                    model = uc.fix_interval_overlap(model, y, w, d, interval_overlap, output_directory)

                # Solve model
                model, status_mip = uc.solve_model(model)

                if status_mip['Solver'][0]['Termination condition'].key != 'optimal':
                    break_flag = True
                    break

                # Fix binary variables
                model = uc.fix_binary_variables(model)

                # Re-solve to obtain prices
                model, status_lp = uc.solve_model(model)

                if status_lp['Solver'][0]['Termination condition'].key != 'optimal':
                    break_flag = True
                    break

                # Save solution
                solution = uc.save_solution(model, y, w, d, output_directory)

                # Unfix binary variables
                model = uc.unfix_binary_variables(model)

                if (d == 7) and (w <= 51):
                    # Get cumulative scheme revenue
                    cumulative_revenue = mpc.get_cumulative_scheme_revenue(y, w + 1)

                    # Get generator energy forecast for following calibration intervals
                    energy_forecast = forecast.get_energy_forecast_persistence()

                    # Get updated baselines
                    mpc_results = mpc.run_baseline_updater(mpc_model, y, w + 1, baseline_start=model.BASELINE[1].value,
                                                           revenue_start=cumulative_revenue, revenue_target=0,
                                                           revenue_floor=float(-1e6),
                                                           permit_price=model.PERMIT_PRICE.value,
                                                           energy_forecast=energy_forecast)

                    # Save MPC results
                    mpc.save_results(y, w + 1, mpc_results)

                    # Update baseline (starting at beginning of following day)
                    for h in [t for t in model.T if t > interval_overlap]:
                        model.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                    # Fix variables up until end of day (beginning of overlap period for next day)
                    model = uc.fix_interval(model, start=1, end=interval_overlap)

                    # Re-run model (MILP)
                    model, status_mip = uc.solve_model(model)

                    if status_mip['Solver'][0]['Termination condition'].key != 'optimal':
                        break_flag = True
                        break

                    # Fix all binary variables
                    model = uc.fix_binary_variables(model)

                    # Re-run model (LP) to solve for prices
                    model, status_lp = uc.solve_model(model)

                    if status_lp['Solver'][0]['Termination condition'].key != 'optimal':
                        break_flag = True
                        break

                    # Save solution (updates previously saved solution for this interval)
                    solution = uc.save_solution(model, y, w, d, output_directory)

                    # Unfix binary variables
                    model = uc.unfix_binary_variables(model)

                    # Unfix remaining variables
                    model = uc.unfix_interval(model, start=1, end=interval_overlap)

                    # All intervals = baseline obtained from MPC model
                    for h in model.T:
                        model.BASELINE[h] = float(mpc_results['baseline_trajectory'][1])

                # Update rolling window counter
                window += 1