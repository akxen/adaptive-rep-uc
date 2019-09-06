"""Model components common to UC and MPC models"""

from pyomo.environ import *

from data import ModelData


class CommonComponents:
    def __init__(self):
        self.data = ModelData()

    def define_parameters(self, m):
        """Common parameters"""

        def emissions_intensity_rule(_m, g):
            """Emissions intensity (tCO2/MWh)"""

            return float(self.data.generators.loc[g, 'EMISSIONS'])

        # Emissions intensities for all generators
        m.EMISSIONS_RATE = Param(m.G, rule=emissions_intensity_rule)

        return m

