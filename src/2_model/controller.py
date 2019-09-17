"""File used to control rolling window algorithm"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))

from base.uc import UnitCommitment


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Model parameters
    years = [2017]
    weeks = range(1, 53)
    days = range(1, 8)
    overlap = 17

    # Initialise object used to construct model
    uc = UnitCommitment()

    # Construct model object
    model = uc.construct_model()

    # Counter for model windows
    window = 1

    # r = pd.read_pickle(os.path.join(output_directory, f'interval_{2017}_{1}_{5}.pickle'))
    break_flag = False

    for year in years:
        if break_flag:
            break

        for week in weeks:
            if break_flag:
                break

            for day in days:
                print(f'Running window {window}: year={year}, week={week}, day={day}')

                # Update model parameters for a given day
                model = uc.update_parameters(model, year, week, day)

                if window != 1:
                    # Fix interval start using solution from previous window
                    model = uc.fix_interval_overlap(model, year, week, day, overlap, output_directory)

                # Update policy parameters (weekly)
                if day == 1:
                    # Run MPC algorithm

                    # Update policy parameters
                    pass

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
                solution = uc.save_solution(model, year, week, day, output_directory)

                # Unfix binary variables
                model = uc.unfix_binary_variables(model)

                # Update rolling window counter
                window += 1
