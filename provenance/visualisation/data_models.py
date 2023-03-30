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
from uuid import UUID, uuid4
from typing import Literal
from fairgraph.base import KGProxy
from fairgraph.utility import as_list

import fairgraph.openminds.computation as omcmp
import fairgraph.openminds.core as omcore

from ..common.data_models import (
    Computation, ComputationPatch, Status, Person, ResourceUsage, LaunchConfiguration,
    ComputationalEnvironment, File, SoftwareVersion, ACTION_STATUS_TYPES, status_name_map
)
from ..common.utils import collab_id_from_space

logger = logging.getLogger("ebrains-prov-api")


class Visualisation(Computation):
    """Record of a data analysis"""
    kg_cls = omcmp.Visualization

    type: Literal["visualization"]

    @classmethod
    def from_kg_object(cls, visualization_object, client):
        if isinstance(visualization_object, KGProxy):
            visualization_object = visualization_object.resolve(client, scope="any")
        assert isinstance(visualization_object, omcmp.Visualization)
        obj = visualization_object.resolve(client, scope="any")
        inputs = []
        for input in as_list(obj.inputs):
            if isinstance(input, KGProxy):
                input = input.resolve(client, scope="any")
            if isinstance(input, (omcore.File, omcmp.LocalFile)):
                inputs.append(File.from_kg_object(input, client))
            elif isinstance(input, omcore.SoftwareVersion):
                inputs.append(SoftwareVersion.from_kg_object(input, client))
            else:
                raise TypeError(f"unexpected object type in inputs: {type(input)}")
        return cls(
            id=client.uuid_from_uri(obj.id),  # just obj.uuid, no?
            type=cls.__fields__["type"].type_.__args__[0],  # "data analysis"
            description=visualization_object.description,
            input=inputs,
            output=[File.from_kg_object(outp, client) for outp in as_list(obj.outputs)],
            environment=ComputationalEnvironment.from_kg_object(obj.environment, client),
            launch_config=LaunchConfiguration.from_kg_object(obj.launch_configuration, client),
            start_time=obj.start_time,
            end_time=obj.end_time,
            started_by=Person.from_kg_object(obj.started_by, client),
            status=getattr(Status, status_name_map[obj.status.resolve(client).name]),
            resource_usage=[ResourceUsage.from_kg_object(ru, client) for ru in as_list(obj.resource_usages)],
            tags=as_list(obj.tags),
            recipe_id=visualization_object.recipe.uuid if visualization_object.recipe else None,
            project_id=collab_id_from_space(visualization_object.space)
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
        resource_usage = [ru.to_kg_object(client) for ru in self.resource_usage]
        recipe_obj = None
        if self.recipe_id:
            recipe_obj = omcmp.WorkflowRecipeVersion.from_uuid(str(self.recipe_id), client, scope="any")
        if self.id is None:
            self.id = uuid4()
        obj = self.__class__.kg_cls(
            id=client.uri_from_uuid(str(self.id)),
            lookup_label=f"Visualisation by {started_by.full_name} on {self.start_time.isoformat()} [{self.id.hex[:7]}]",
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


class VisualisationPatch(ComputationPatch):
    """Correction of or update to a record of a data analysis"""

    def apply_to_kg_object(self, visualization_object, client):
        obj = visualization_object
        update_label = False
        if self.input:
            obj.inputs = [inp.to_kg_object(client) for inp in self.input]
        if self.output:
            obj.outputs = [outp.to_kg_object(client) for outp in self.output]
        if self.environment:
            obj.environment = self.environment.to_kg_object(client)
        if self.launch_config:
            obj.launch_configuration = self.launch_config.to_kg_object(client)
        if self.start_time:
            obj.started_at_time = self.start_time
            update_label = True
        if self.end_time:
            obj.ended_at_time = self.end_time
        if self.started_by:
            obj.started_by = self.started_by.to_kg_object(client)
            update_label = True
        if self.status:
            obj.status = ACTION_STATUS_TYPES[self.status.value]
        if self.resource_usage:
            obj.resource_usages = [ru.to_kg_object(client) for ru in self.resource_usage]
        if self.tags:
            obj.tags = as_list(self.tags)
        if update_label:
            obj.lookup_label = f"Visualisation by {obj.started_by.full_name} on {obj.started_at_time.isoformat()} [{obj.id.hex[:7]}]"
        return obj