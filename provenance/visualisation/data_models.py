"""
docstring goes here
"""

"""
   Copyright 2021 CNRS

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import logging
from fairgraph.openminds.computation import Visualization as KGVisualization
from ..dataanalysis.data_models import DataAnalysis, DataAnalysisPatch

logger = logging.getLogger("ebrains-prov-api")


class Visualisation(DataAnalysis):
    """Record of a visualisation"""
    kg_cls = KGVisualization


class VisualisationPatch(DataAnalysisPatch):
    """Correction of or update to a record of a visualisation"""
    kg_cls = KGVisualization
