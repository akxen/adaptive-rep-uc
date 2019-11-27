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
    # for name, parameters in case_parameters.items():
    for name in ['bau',
                 'renewables_eligibility',
                 'revenue_floor',
                 'revenue_target_1_ci',
                 'revenue_target_3_ci',
                 'anticipated_emissions_intensity_shock_1_ci',
                 'anticipated_emissions_intensity_shock_3_ci',
                 'unanticipated_emissions_intensity_shock_1_ci',
                 'unanticipated_emissions_intensity_shock_3_ci',
                 'multi_scenario_forecast',
                 'persistence_forecast'
                 ]:
        parameters = case_parameters[name]
        print(f'Running case: {name}')
        print(parameters)

        # Create directory if it doesn't exist and cleanup files
        cases.create_case_directory(parameters['output_dir'])
        cases.cleanup_directory(parameters['output_dir'])

        # Run case
        cases.run_case(parameters)
