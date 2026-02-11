"""
docstring goes here
"""

"""
   Copyright 2022 CNRS

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
from uuid import UUID
from typing import ClassVar, Union, Literal, List
from fairgraph import KGObject
from fairgraph.openminds.computation import Optimization as KGOptimization

from ..common.data_models import (
    File,
    Computation,
    ComputationPatch,
    SoftwareVersion,
    ModelVersionReference
)

logger = logging.getLogger("ebrains-prov-api")


class Optimisation(Computation):
    """Record of an optimization"""
    kg_cls: ClassVar[KGObject] = KGOptimization

    input: List[Union[File, ModelVersionReference, SoftwareVersion]]
    output: List[Union[File, ModelVersionReference]]
    type: Literal["optimization"]


class OptimisationPatch(ComputationPatch):
    """Correction of or update to a record of an optimisation"""
    input: List[Union[File, ModelVersionReference, SoftwareVersion]] = None
    output: List[Union[File, ModelVersionReference]] = None
