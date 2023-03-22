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
from enum import Enum
from uuid import UUID, uuid4
from typing import List, Union, Literal

from pydantic import Field

from fairgraph.base import KGProxy
from fairgraph.utility import as_list
import fairgraph.openminds.computation as omcmp
import fairgraph.openminds.core as omcore

from ..common.data_models import (
    File,
    Computation,
    ComputationPatch,
    SoftwareVersion,
    ModelVersionReference,
    Status,
    Person,
    ResourceUsage,
    LaunchConfiguration,
    ComputationalEnvironment,
    ACTION_STATUS_TYPES,
    status_name_map,
    ComputationType
)


from .examples import EXAMPLES


logger = logging.getLogger("ebrains-prov-api")


class Simulator(str, Enum):
    """docstring for Simulator goes here"""
    nest="NEST"
    neuron="NEURON"
    brian="Brian"
    arbor="Arbor"
    spinnaker="SpiNNaker"
    brainscales="BrainScaleS"


class Simulation(Computation):
    """Record of a numerical simulation"""
    kg_cls = omcmp.Simulation

    input: List[Union[File, ModelVersionReference, SoftwareVersion]] = Field(...,
        description="Inputs to this simulation (models, data files, configuration files and/or code)")
    type: Literal["simulation"]

    # informed_by: "SimulationNew" = None

    class Config:
        schema_extra = EXAMPLES["Simulation"]

    @classmethod
    def from_kg_object(cls, simulation_object, client):
        if isinstance(simulation_object, KGProxy):
            simulation_object = simulation_object.resolve(client, scope="any")
        assert isinstance(simulation_object, omcmp.Simulation)
        inputs = []
        for obj in as_list(simulation_object.inputs):
            if isinstance(obj, KGProxy):
                obj = obj.resolve(client, scope="any")
            if isinstance(obj, (omcore.File, omcmp.LocalFile)):
                inputs.append(File.from_kg_object(obj, client))
            elif isinstance(obj, omcore.SoftwareVersion):
                inputs.append(SoftwareVersion.from_kg_object(obj, client))
            elif isinstance(obj, omcore.ModelVersion):
                inputs.append(ModelVersionReference.from_kg_object(obj, client))
            else:
                raise TypeError(f"unexpected object type in inputs: {type(obj)}")
        return cls(
            id=client.uuid_from_uri(simulation_object.id),
            type=cls.__fields__["type"].type_.__args__[0],
            input=inputs,
            output=[File.from_kg_object(obj, client) for obj in as_list(simulation_object.outputs)],
            environment=ComputationalEnvironment.from_kg_object(simulation_object.environment, client),
            launch_config=LaunchConfiguration.from_kg_object(simulation_object.launch_configuration, client),
            start_time=simulation_object.start_time,
            end_time=simulation_object.end_time,
            started_by=Person.from_kg_object(simulation_object.started_by, client),
            status=getattr(Status, status_name_map[simulation_object.status.resolve(client).name]),
            resource_usage=[ResourceUsage.from_kg_object(obj, client) for obj in as_list(simulation_object.resource_usages)],
            tags=as_list(simulation_object.tags),
            recipe_id=simulation_object.recipe.uuid if simulation_object.recipe else None
        )

    def to_kg_object(self, client):
        if self.started_by:
            started_by = self.started_by.to_kg_object(client)
        else:
            started_by = omcore.Person.me(client)  # todo
        inputs = [inp.to_kg_object(client) for inp in self.input]
        outputs = [outp.to_kg_object(client) for outp in self.output]
        environment = self.environment.to_kg_object(client)
        launch_configuration = self.launch_config.to_kg_object(client)
        if self.resource_usage:
            resource_usage = [ru.to_kg_object(client) for ru in self.resource_usage]
        else:
            resource_usage = None
        recipe_obj = None
        if self.recipe_id:
            recipe_obj = omcmp.WorkflowRecipeVersion.from_uuid(str(self.recipe_id), client, scope="any")
        if self.id is None:
            self.id = uuid4()
        obj = self.__class__.kg_cls(
            id=client.uri_from_uuid(str(self.id)),
            lookup_label=f"Simulation run by {started_by.full_name} on {self.start_time.isoformat()} [{self.id.hex[:7]}]",
            inputs=inputs,
            outputs=outputs,
            environment=environment,
            launch_configuration=launch_configuration,
            start_time=self.start_time,
            end_time=self.end_time,
            started_by=started_by,
            #was_informed_by= # todo
            status=ACTION_STATUS_TYPES[self.status.value],
            resource_usages=resource_usage,
            tags=self.tags,
            recipe=recipe_obj
        )
        return obj


class SimulationPatch(ComputationPatch):
    """Correction of or update to a record of a numerical simulation"""

    input: List[Union[File, ModelVersionReference, SoftwareVersion]] = None
    # informed_by: "Simulation" = None

    class Config:
        schema_extra = {
            "example": {
                "end_time": "2021-05-28T16:32:58.597Z",
                "status": "failed",
                "resource_usage": [
                    {
                        "value": 1017.3,
                        "units": "core-hour"
                    }
                ],
                "tags": [
                    "core-dump"
                ]
            }
        }
