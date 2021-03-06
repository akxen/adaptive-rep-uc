"""File used to control rolling window algorithm"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))

from base.cases import ModelCases

if __name__ == '__main__':
    # Directory for output files
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Object used to construct and run model cases
    cases = ModelCases()

    # Generate case parameters for given year and week ranges
    case_parameters = cases.generate_cases(years=[2018], weeks=range(1, 53), output_dir=output_directory)

    # Run all cases
    for name, parameters in case_parameters.items():

    # Run selected cases - comment line above and uncomment following lines to run selected cases
    # names = ([f'anticipated_emissions_intensity_shock_{c}_ci' for c in [1, 3, 6]]
    #          + [f'unanticipated_emissions_intensity_shock_{c}_ci' for c in [1, 3, 6]]
    #          + [f'revenue_target_{c}_ci' for c in [1, 3, 6]])
    # for name in names:
    #     parameters = case_parameters[name]

        print(f'Running case: {name}')
        print(parameters)

        # Create directory if it doesn't exist and cleanup files
        # cases.create_case_directory(parameters['output_dir'])
        # cases.cleanup_directory(parameters['output_dir'])

        # Run case
        cases.run_case(parameters)
