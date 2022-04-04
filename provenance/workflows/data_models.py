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

from typing import List, Union
from uuid import UUID
from typing_extensions import Annotated

from pydantic import BaseModel, Field

from fairgraph.base_v3 import KGProxy
import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp
from ..common.data_models import Person
from ..simulation.data_models import Simulation
from ..dataanalysis.data_models import DataAnalysis
from ..visualisation.data_models import Visualisation
from ..optimisation.data_models import Optimisation


class WorkflowRecipe(BaseModel):
    pass


# see https://stackoverflow.com/questions/70914419/how-to-get-pydantic-to-discriminate-on-a-field-within-listuniontypea-typeb
_Computation = Annotated[
    Union[Simulation, DataAnalysis, Visualisation, Optimisation],
    Field(discriminator="type")
]


class WorkflowExecution(BaseModel):
    kg_cls = omcmp.WorkflowExecution

    id: UUID = None
    stages: List[_Computation] = Field(
        ...,
        description="A workflow record is specified by the list of computations that were carried out. "
                    "The sequence of computations is inferred from the inputs, outputs and start times of each stage."
    )
    started_by: Person = None
    recipe_id: UUID = None


    @classmethod
    def from_kg_object(cls, workflow_execution_object, client):
        weo = workflow_execution_object
        cls_map = {
            omcmp.DataAnalysis: DataAnalysis,
            omcmp.Visualization: Visualisation,
            omcmp.Simulation: Simulation,
            omcmp.Optimization: Optimisation
        }
        def get_class(obj):
            if isinstance(obj, KGProxy):
                return obj.cls
            else:
                return obj.__class__
        stages = [
            cls_map[get_class(stage)].from_kg_object(stage, client)
            for stage in weo.stages
        ]
        return cls(
            id=weo.uuid,
            stages=stages,
            recipe_id=weo.recipe.uuid if weo.recipe else None,
            started_by=Person.from_kg_object(weo.started_by, client) if weo.started_by else None
        )

    def to_kg_object(self, client):
        if self.started_by:
            started_by = self.started_by.to_kg_object(client)
            for stage in self.stages:
                if stage.started_by is None:
                    stage.started_by = self.started_by
        else:
            started_by = omcore.Person.me(client)
        stages = [stage.to_kg_object(client) for stage in self.stages]
        if self.recipe_id:
            recipe = omcmp.WorkflowRecipeVersion.from_id(str(self.recipe_id), client, scope="in progress")  # todo: also search scope="released"
        else:
            recipe = None
        # todo: error message if recipe is not found
        obj = self.__class__.kg_cls(
            id=client.uri_from_uuid(self.id),
            started_by=started_by,
            recipe=recipe,
            stages=stages
        )
        return obj
