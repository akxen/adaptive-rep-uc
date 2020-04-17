"""Create tables"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_model', 'base'))

import pandas as pd

from processing import Results


class CreateTables:
    def __init__(self, output_dir):
        # Object used to process results
        self.results = Results()

        # Output directory for tables
        self.output_dir = output_dir

    @staticmethod
    def load_cache(directory, filename):
        """Load cached results"""

        with open(os.path.join(directory, filename), 'rb') as f:
            results = pickle.load(f)

        return results

    @staticmethod
    def save(results, directory, filename):
        """Save results"""

        with open(os.path.join(directory, filename), 'wb') as f:
            pickle.dump(results, f)

    def get_summary_table_data(self, use_cache=False, save=True):
        """Get data for table summarising results from all runs"""

        filename = 'summary_table_data.pickle'

        # Load cached file if specified and return (faster loading)
        if use_cache:
            results = self.load_cache(self.output_dir, filename)
            return results

        # Container for results
        results = {}

        cases = ([f'{i}_calibration_intervals' for i in range(1, 7)]
                 + [f'revenue_target_{i}_ci' for i in [1, 3, 6]]
                 + [f'anticipated_emissions_intensity_shock_{i}_ci' for i in [1, 3, 6]]
                 + [f'unanticipated_emissions_intensity_shock_{i}_ci' for i in [1, 3, 6]]
                 + ['bau', 'carbon_tax', 'renewables_eligibility', 'revenue_floor'])

        for c in cases:
            # Extract baselines, scheme revenue, and prices
            baselines, revenue = self.results.get_baselines_and_revenue(c)
            prices, _ = self.results.get_week_prices(c)

            # Get emissions
            emissions = self.results.get_day_emissions(c)

            # Case parameters
            parameters = self.results.get_case_parameters(c)

            # Combine results into single dictionary
            results[c] = {'baselines': baselines, 'revenue': revenue, 'prices': prices, 'emissions': emissions,
                          'parameters': parameters}

        # Save results
        if save:
            self.save(results, self.output_dir, filename)

        return results

    @staticmethod
    def get_case_summary(results, case):
        """Extract and format results for each case"""

        # Results for case
        c_r = results[case]

        # Mappings between case names and index to be used in table
        case_map = {**{f'{i}_calibration_intervals': ('revenue neutral', f'{i} CI') for i in range(1, 7)},
                    **{f'unanticipated_emissions_intensity_shock_{i}_ci': ('unanticipated shock', f'{i} CI') for i in
                       [1, 3, 6]},
                    **{f'anticipated_emissions_intensity_shock_{i}_ci': ('anticipated shock', f'{i} CI') for i in
                       [1, 3, 6]},
                    **{f'revenue_target_{i}_ci': ('revenue target', f'{i} CI') for i in [1, 3, 6]},
                    'revenue_floor': ('Revenue floor', '3 CI'),
                    'bau': ('Benchmark', 'BAU'),
                    'carbon_tax': ('Benchmark', 'Carbon tax'),
                    'renewables_eligibility': ('Renewables eligibility', '3 CI'),
                    }

        # Add dagger for table notes to denote that CI = calibration intervals
        case_map['1_calibration_intervals'] = ('revenue neutral', '1 CI$^{\dagger}$')

        def pad_decimal_places(x, places):
            """Zero pad to a given number of decimal places"""

            # Split original number (should already be rounded)
            integer, decimal = str(x).split('.')

            # Check if already have required number of decimal places
            if len(decimal) == places:
                return f'{integer}.{decimal}'

            # Number of zeros that must be padded
            pad = places - len(decimal)

            # New zero padded decimal number
            number = f'{integer}.{decimal}' + ''.join(['0'] * pad)

            return number

        # Extract price data
        prices = pd.DataFrame({case_map[case]: {'mean': c_r['prices'].mean(), 'std': c_r['prices'].std(),
                                                'min': c_r['prices'].min(), 'max': c_r['prices'].max()}}).T
        prices = prices[['mean', 'std', 'min', 'max']]
        prices = prices.round(2).applymap(lambda x: pad_decimal_places(x, 2))

        # Rename columns and index
        prices.columns = pd.MultiIndex.from_product([['Price (\$/MWh)'], prices.columns])

        # Extract emissions data
        emissions = pd.DataFrame({case_map[case]: {'total': c_r['emissions'].sum()}}).T.div(1e6)
        emissions = emissions.round(2).applymap(lambda x: pad_decimal_places(x, 2))

        # Rename columns and index
        emissions.columns = pd.MultiIndex.from_product([['Emissions (MtCO$_{2}$)'], emissions.columns])

        # Get revenue imbalance
        revenue_target = pd.DataFrame(c_r['parameters']['revenue_target']).stack().reorder_levels([1, 0])

        if case not in ['bau', 'carbon_tax']:
            revenue_difference = c_r['revenue'].subtract(revenue_target)

            # Compute statistics for revenue imbalance
            revenue = pd.DataFrame({case_map[case]: {'mean': revenue_difference.mean(), 'std': revenue_difference.std(),
                                                     'min': revenue_difference.min(),
                                                     'max': revenue_difference.max()}}).T
            revenue = revenue[['mean', 'std', 'min', 'max']].div(1e6)
            revenue = revenue.round(2).applymap(lambda x: pad_decimal_places(x, 2))

            # Extract baseline data
            baselines = pd.DataFrame({case_map[case]: {'mean': c_r['baselines'].mean(), 'std': c_r['baselines'].std(),
                                                       'min': c_r['baselines'].min(), 'max': c_r['baselines'].max()}}).T
            baselines = baselines[['mean', 'std', 'min', 'max']]
            baselines = baselines.round(3).applymap(lambda x: pad_decimal_places(x, 3))

        else:
            # Placeholders for business as usual case and carbon tax
            revenue = pd.DataFrame({case_map[case]: {'mean': '-', 'std': '-', 'min': '-', 'max': '-'}}).T
            revenue = revenue[['mean', 'std', 'min', 'max']]

            # Baseline placeholders
            baselines = pd.DataFrame({case_map[case]: {'mean': '-', 'std': '-', 'min': '-', 'max': '-'}}).T
            baselines = baselines[['mean', 'std', 'min', 'max']]

        # Rename columns
        baselines.columns = pd.MultiIndex.from_product([['Baseline (tCO$_{2}$/MWh)'], baselines.columns])
        revenue.columns = pd.MultiIndex.from_product([['Revenue imbalance (M\$)'], revenue.columns])

        # Combine all statistics for a given case into a single DataFrame
        case_summary = pd.concat([baselines, revenue, prices, emissions], axis=1)

        return case_summary

    def process_all_cases(self, results, table_name):
        """Extract results for all cases"""

        # Container for case results
        case_summary_results = []

        # Names of all cases to extract data for
        all_cases = (['bau', 'carbon_tax']
                     + [f'{i}_calibration_intervals' for i in range(1, 7)]
                     + [f'revenue_target_{i}_ci' for i in [1, 3, 6]]
                     + [f'anticipated_emissions_intensity_shock_{i}_ci' for i in [1, 3, 6]]
                     + [f'unanticipated_emissions_intensity_shock_{i}_ci' for i in [1, 3, 6]]
                     + ['renewables_eligibility', 'revenue_floor'])

        # Extract and format data for all cases
        for c in all_cases:
            # Get results for each case
            print('Processing', c)
            case_results = self.get_case_summary(results, c)

            # Append results to main container
            case_summary_results.append(case_results)

        # Concatenate into single DataFrame
        df_results = pd.concat(case_summary_results)

        # Save in latex format
        df_results.to_latex(os.path.join(self.output_dir, table_name), column_format='ll*{13}{r}', escape=False,
                            multicolumn_format='c')

    @staticmethod
    def format_summary_table(table_name):
        """Apply formatting to summary table"""

        # Load table
        with open(os.path.join(tables.output_dir, table_name), 'r') as f:
            tab = f.read()

        # Apply formatting specific to table
        tab_edit = tab.replace('Benchmark', r'\multicolumn{2}{l}{Benchmark} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace('revenue neutral',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Revenue neutral} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace('revenue target',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Revenue target} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace('unanticipated shock',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Unanticipated shock} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace(r'anticipated shock ',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Anticipated shock} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace(r'Renewables eligibility ',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Renewables eligibility} &&&&&&&&&&&&&\\')
        tab_edit = tab_edit.replace(r'Revenue floor ',
                                    r'&&&&&&&&&&&&&& \\ \multicolumn{2}{l}{Revenue floor} &&&&&&&&&&&&&\\')

        # Save new table
        formatted_table_name = f"{table_name.split('.')[0]}_formatted.tex"
        with open(os.path.join(tables.output_dir, formatted_table_name), 'w') as f:
            f.write(tab_edit)

        return tab, tab_edit

    def create_summary_table(self):
        """Create table summarising results from all cases and apply formatting"""

        # Name of summary table
        table_name = 'summary_table.tex'

        # Get data used to construct summary table
        try:
            r = self.get_summary_table_data(use_cache=True)
        except:
            r = self.get_summary_table_data(use_cache=False)

        # Construct table
        self.process_all_cases(results=r, table_name=table_name)

        # Apply formatting
        self.format_summary_table(table_name)


if __name__ == '__main__':
    tables_output_directory = os.path.join(os.path.dirname(__file__), 'output', 'tables')

    tables = CreateTables(tables_output_directory)

    # Get results to be used in results summary table
    r9 = tables.get_summary_table_data(use_cache=False)
    tables.create_summary_table()
