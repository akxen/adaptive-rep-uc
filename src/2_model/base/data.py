"""Collect model input data"""

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelData:

    # Directory containing core data files
    data_dir: str = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'data')

    # Directory containing dispatch and demand traces
    traces_dir: str = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '1_traces', 'output')

    def __post_init__(self):
        """Append data objects to class object"""

        # Name of NTNDP file
        self.ntndp_filename = '2016 Planning Studies - Additional Modelling Data and Assumptions summary.xlsm'

        # Minimum reserve levels for each NEM region
        self.minimum_reserve_levels = self.get_minimum_reserve_levels()

        # Storage unit properties
        self.battery_properties = self.get_ntndp_battery_properties()

        # Generator and storage data Data
        self.generators = self.get_generator_data()
        self.storage = self.get_storage_unit_data()

        # NEM zones and regions
        self.nem_zones = self.get_nem_zones()
        self.nem_regions = self.get_nem_regions()

        # DUIDs for different units
        self.scheduled_duids = self.get_scheduled_unit_duids()
        self.semi_scheduled_duids = self.get_semi_scheduled_unit_duids()
        self.solar_duids = self.get_solar_unit_duids()
        self.wind_duids = self.get_wind_unit_duids()
        self.thermal_duids = self.get_thermal_unit_duids()
        self.hydro_duids = self.get_hydro_unit_duids()
        self.storage_duids = self.get_storage_unit_duids()
        self.slow_start_duids = self.get_slow_start_thermal_generator_ids()
        self.quick_start_duids = self.get_quick_start_thermal_generator_ids()

        # Mapping between NEM regions and zones
        self.nem_region_zone_map = self.get_nem_region_zone_map()

        # Mapping between DUIDs and NEM zones
        self.duid_zone_map = self.get_duid_zone_map()

        # Network incidence matrix
        self.network_incidence_matrix = self.get_network_incidence_matrix()

        # Links between adjacent NEM regions
        self.links = self.get_network_links()

        # Links that have constrained flows
        self.links_constrained = self.get_link_powerflow_limits().keys()

        # Power flow limits for constrained links
        self.powerflow_limits = self.get_link_powerflow_limits()

        # Demand traces for each day
        self.demand = pd.read_pickle(os.path.join(self.traces_dir, 'demand_42.pickle'))

        # Dispatch traces for each day
        self.dispatch = pd.read_pickle(os.path.join(self.traces_dir, 'dispatch_42.pickle'))

    def get_minimum_reserve_levels(self):
        """Minimum reserve levels for each NEM region"""

        # Minimum reserve levels for each NEM region
        df = pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename), sheet_name='MRL', skiprows=1)

        # Rename columns and set index as NEM region
        df_o = (df.rename(columns={'Region': 'NEM_REGION', 'Minimum Reserve Level (MW)': 'MINIMUM_RESERVE_LEVEL'})
                .set_index('NEM_REGION'))

        # Keep latest minimum reserve values (SA1 has multiple MRL values with different start dates)
        df_o = df_o.loc[~df_o.index.duplicated(keep='last'), 'MINIMUM_RESERVE_LEVEL'].to_dict()

        return df_o

    def get_ntndp_battery_properties(self):
        """Load battery properties from NTNDP database"""

        # Battery properties from NTNDP worksheet
        df = (pd.read_excel(os.path.join(self.data_dir, 'files', self.ntndp_filename),
                            sheet_name='Battery Properties', skiprows=1)
              .rename(columns={'Battery': 'STORAGE_ID'}).set_index('STORAGE_ID'))

        return df

    def get_generator_data(self):
        """Load generator data"""

        # Path to generator data file
        path = os.path.join(self.data_dir, 'files', 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603',
                            'generators', 'generators.csv')

        # Load generator data into DataFrame
        df = pd.read_csv(path, index_col='DUID')

        return df

    def get_nem_zones(self):
        """Get tuple of unique NEM zones"""

        # Load generator information
        df = self.get_generator_data()

        # Extract nem zones from existing generators dataset
        zones = tuple(df.loc[:, 'NEM_ZONE'].unique())

        # There should be 16 zones
        assert len(zones) == 16, 'Unexpected number of NEM zones'

        return zones

    def get_nem_regions(self):
        """Get tuple of unique NEM regions"""

        # Load generator information
        df = self.get_generator_data()

        # Extract nem regions from existing generators dataset
        regions = tuple(df.loc[:, 'NEM_REGION'].unique())

        # There should be 5 NEM regions
        assert len(regions) == 5, 'Unexpected number of NEM regions'

        return regions

    def get_nem_region_zone_map(self):
        """Construct mapping between NEM regions and the zones belonging to those regions"""

        # Load generator information
        df = self.get_generator_data()

        # Map between NEM regions and zones
        region_zone_map = (df[['NEM_REGION', 'NEM_ZONE']] .drop_duplicates(subset=['NEM_REGION', 'NEM_ZONE'])
                           .groupby('NEM_REGION')['NEM_ZONE'].apply(lambda x: tuple(x))).to_dict()

        return region_zone_map

    def get_duid_zone_map(self):
        """Get mapping between DUIDs and NEM zones"""

        # Load generator and storage information
        df_g = self.get_generator_data()
        df_s = self.get_storage_unit_data()

        # Get mapping between DUIDs and zones for generators and storage units
        generator_map = df_g.loc[:, 'NEM_ZONE'].to_dict()
        storage_map = df_s.loc[:, 'NEM_ZONE'].to_dict()

        # Combine dictionaries
        zone_map = {**generator_map, **storage_map}

        return zone_map

    def get_thermal_unit_duids(self):
        """Get thermal unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Thermal DUIDs
        thermal_duids = df.loc[df['FUEL_CAT'] == 'Fossil'].index

        return thermal_duids

    def get_solar_unit_duids(self):
        """Get solar unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Solar DUIDs
        solar_duids = df.loc[df['FUEL_CAT'] == 'Solar'].index

        return solar_duids

    def get_wind_unit_duids(self):
        """Get wind unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Wind DUIDs
        wind_duids = df.loc[df['FUEL_CAT'] == 'Wind'].index

        return wind_duids

    def get_hydro_unit_duids(self):
        """Get hydro unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Hydro DUIDs
        hydro_duids = df.loc[df['FUEL_CAT'] == 'Hydro'].index

        return hydro_duids

    def get_scheduled_unit_duids(self):
        """Get all scheduled unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Thermal DUIDs
        scheduled_duids = df.loc[df['SCHEDULE_TYPE'] == 'SCHEDULED'].index

        return scheduled_duids

    def get_semi_scheduled_unit_duids(self):
        """Get all scheduled unit DUIDs"""

        # Load generator information
        df = self.get_generator_data()

        # Semi scheduled DUIDs
        semi_scheduled_duids = df.loc[df['SCHEDULE_TYPE'] == 'SEMI-SCHEDULED'].index

        return semi_scheduled_duids

    def get_network_incidence_matrix(self):
        """Construct network incidence matrix"""

        # All NEM zones:
        zones = self.get_nem_zones()

        # Links connecting different zones. First zone is 'from' zone second is 'to' zone
        links = ['NQ-CQ', 'CQ-SEQ', 'CQ-SWQ', 'SWQ-SEQ', 'SEQ-NNS',
                 'SWQ-NNS', 'NNS-NCEN', 'NCEN-CAN', 'CAN-SWNSW',
                 'CAN-NVIC', 'SWNSW-NVIC', 'LV-MEL', 'NVIC-MEL',
                 'TAS-LV', 'MEL-CVIC', 'SWNSW-CVIC', 'CVIC-NSA',
                 'MEL-SESA', 'SESA-ADE', 'NSA-ADE']

        # Initialise empty matrix with NEM zones as row and column labels
        incidence_matrix = pd.DataFrame(index=links, columns=zones, data=0)

        # Assign values to 'from' and 'to' zones. +1 is a 'from' zone, -1 is a 'to' zone
        for link in links:
            # Get from and to zones
            from_zone, to_zone = link.split('-')

            # Set from zone element to 1
            incidence_matrix.loc[link, from_zone] = 1

            # Set to zone element to -1
            incidence_matrix.loc[link, to_zone] = -1

        return incidence_matrix

    def get_network_links(self):
        """Links connecting adjacent NEM zones"""

        return self.get_network_incidence_matrix().index

    @staticmethod
    def get_link_powerflow_limits():
        """Max forward and reverse power flow over links between zones"""

        # Limits for interconnectors composed of single branches
        interconnector_limits = {'SEQ-NNS': {'forward': 210, 'reverse': 107},  # Terranora
                                 'SWQ-NNS': {'forward': 1078, 'reverse': 600},  # QNI
                                 'TAS-LV': {'forward': 594, 'reverse': 478},  # Basslink
                                 'MEL-SESA': {'forward': 600, 'reverse': 500},  # Heywood
                                 'CVIC-NSA': {'forward': 220, 'reverse': 200},  # Murraylink
                                 }

        return interconnector_limits

    def get_slow_start_thermal_generator_ids(self):
        """
        Get IDs for existing and candidate slow start unit

        A generator is classified as 'slow' if it cannot reach its
        minimum dispatchable power output in one interval (e.g. 1 hour).

        Note: A generator's classification of 'quick' or 'slow' depends on its
        minimum dispatchable output level and ramp-rate. For candidate units
        the minimum dispatchable output level is a function of the maximum
        output level, and so is variable. As this level is not known ex ante,
        all candidate thermal generators are assumed to operate the same way
        as quick start units (i.e. they can reach their minimum dispatchable
        output level in 1 trading interval (hour)).
        """

        # Load generator information
        df = self.get_generator_data()

        # True if number of hours to ramp to min generator output > 1
        mask_slow_start = df['MIN_GEN'].div(df['RR_STARTUP']).gt(1)

        # Only consider coal and gas units
        mask_technology = df['FUEL_CAT'].isin(['Fossil'])

        # Get IDs for slow start generators
        gen_ids = df.loc[mask_slow_start & mask_technology, :].index

        return gen_ids

    def get_quick_start_thermal_generator_ids(self):
        """
        Get IDs for existing and candidate slow start unit

        Note: A generator is classified as 'quick' if it can reach its
        minimum dispatchable power output in one interval (e.g. 1 hour).
        """

        # Load generator information
        df = self.get_generator_data()

        # Slow start unit IDs - previously identified
        slow_gen_ids = self.get_slow_start_thermal_generator_ids()

        # Filter for slow generator IDs (existing units)
        mask_slow_gen_ids = df.index.isin(slow_gen_ids)

        # Only consider coal and gas units
        mask_existing_technology = df['FUEL_CAT'].isin(['Fossil'])

        # Get IDs for quick start generators
        existing_quick_gen_ids = df.loc[~mask_slow_gen_ids & mask_existing_technology, :].index

        return existing_quick_gen_ids

    def get_storage_unit_data(self):
        """Get storage unit IDs"""

        # Path to generator data file
        path = os.path.join(self.data_dir, 'files', 'NEM Registration and Exemption List.xls')

        # Load generators and scheduled load data into DataFrame
        df = pd.read_excel(path, index_col='DUID', sheet_name='Generators and Scheduled Loads')

        # Rename selected columns to conform with existing generator data DataFrame
        column_map = {'Station Name': 'STATIONNAME', 'Region': 'NEM_REGION', 'Reg Cap (MW)': 'REG_CAP',
                      'Classification': 'SCHEDULE_TYPE', 'Fuel Source - Primary': 'FUEL_TYPE'}
        df = df.rename(columns=column_map)

        # Only retain storage units
        df_s = df.loc[df['FUEL_TYPE'] == 'Battery storage', :].copy()
        df_s.loc[:, 'Max ROC/Min'] = df_s.loc[:, 'Max ROC/Min'].astype(float)
        df_s.loc[:, 'RR_UP'] = df_s.loc[:, 'Max ROC/Min'].mul(60)
        df_s.loc[:, 'RR_DOWN'] = df_s.loc[:, 'Max ROC/Min'].mul(60)
        df_s.loc[:, 'RR_STARTUP'] = df_s.loc[:, 'Max ROC/Min'].mul(60)
        df_s.loc[:, 'RR_SHUTDOWN'] = df_s.loc[:, 'Max ROC/Min'].mul(60)

        # Change schedule type to upper case
        df_s.loc[:, 'SCHEDULE_TYPE'] = df_s.loc[:, 'SCHEDULE_TYPE'].str.upper()

        # NEM zone map
        nem_zone_map = {'BALBG1': 'MEL', 'DALNTH01': 'NSA', 'GANNBG1': 'NVIC', 'HPRG1': 'NSA'}
        df_s['NEM_ZONE'] = df_s.apply(lambda x: nem_zone_map[x.name], axis=1)

        # Default storage parameter assumptions (efficiency from NTNDP database spreadsheet)
        storage_parameters = {'EMISSIONS': float(0), 'MIN_GEN': float(0), 'MIN_ON_TIME': float(0),
                              'MIN_OFF_TIME': float(0),
                              'SU_COST_COLD': float(0), 'SU_COST_WARM': float(0), 'SU_COST_HOT': float(0),
                              'VOM': float(7), 'FUEL_CAT': 'STORAGE',
                              'HEAT_RATE': float(0), 'NL_FUEL_CONS': float(0), 'FC_2016-17': float(0),
                              'EFFICIENCY': 0.92}
        params = pd.Series(storage_parameters)

        # Append storage unit parameters
        df_s = df_s.apply(lambda x: pd.Series({**x.to_dict(), **params}), axis=1)
        df_s.loc[:, 'SRMC_2016-17'] = df_s['VOM']

        # Retain selected columns
        keep_cols = list(column_map.values()) + list(storage_parameters.keys()) + ['NEM_ZONE', 'SRMC_2016-17']
        df_s = df_s.reindex(keep_cols, axis=1)

        return df_s

    def get_storage_unit_duids(self):
        """Get DUIDs for storage units"""

        # Storage unit data
        df = self.get_storage_unit_data()

        return df.index


if __name__ == '__main__':
    # Path to core data directory
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'data')

    # Object used to extract model data from raw files
    dataset = ModelData()

    df_gen = dataset.get_generator_data()
    df_b = dataset.get_ntndp_battery_properties()
    nem_regions = dataset.get_nem_regions()
    nem_region_zone_map = dataset.get_nem_region_zone_map()

    gen_thermal_duids = dataset.get_thermal_unit_duids()
    gen_solar_duids = dataset.get_solar_unit_duids()
    gen_wind_duids = dataset.get_wind_unit_duids()
    gen_hydro_duids = dataset.get_hydro_unit_duids()
    gen_scheduled_duids = dataset.get_scheduled_unit_duids()
    gen_semi_scheduled_duids = dataset.get_semi_scheduled_unit_duids()

    network_incidence_matrix = dataset.get_network_incidence_matrix()

    gen_slow_start = dataset.get_slow_start_thermal_generator_ids()
    gen_quick_start = dataset.get_quick_start_thermal_generator_ids()

    storage = dataset.get_storage_unit_data()
