"""
docstring goes here
"""

"""
   Copyright 2021-2022 CNRS

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

from os import environ, stat_result
from typing import List
from uuid import UUID, uuid4
from datetime import datetime
import logging
from itertools import chain


from fastapi import APIRouter, Depends, Header, Query, HTTPException, status as status_codes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from fairgraph.utility import as_list
import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp

from ..auth.utils import get_kg_client_for_user_account

from .data_models import Visualisation, VisualisationPatch
from ..common.data_models import HardwareSystem, Status, ACTION_STATUS_TYPES
from ..common.utils import create_computation, replace_computation, patch_computation, delete_computation, NotFoundError
from .. import settings


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/visualisations/", response_model=List[Visualisation])
def query_visualisations(
    dataset: UUID = Query(None, description="Return visualisations of this dataset"),
    simulation: UUID = Query(None, description="Return visualisations of results from this simulation"),
    input_data: UUID = Query(None, description="Return visualisations of a given data file or directory containing data files"),
    software: UUID = Query(None, description="Return visualisations that used a specific software version"),
    platform: HardwareSystem = Query(None, description="Return visualisations that ran on this hardware platform"),
    space: str = Query(None, description="Knowledge Graph space to search in"),
    status: Status = Query(None, description="Return visualisations with this status"),
    tags: List[str] = Query(None, description="Return visualisations with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),

):
    """
    Query recorded data visualisations, filtered according to various criteria.

    Where multiple filters are applied, they are combined with AND,
    e.g. software=<UUID for Elephant v0.9.0>&platform=pizdaint returns
    data visualisations that used Elephant v0.9.0 and that ran on Piz Daint.

    The list may contain records of data visualisations that are public, were performed by the logged-in user,
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    filters = {
        "inputs": [],
        "environment": []
    }
    if dataset:
        dataset_obj = omcore.DatasetVersion.from_id(str(dataset), kg_client)
        if dataset_obj is None:
            raise HTTPException(
                status_code=status_codes.HTTP_404_NOT_FOUND,
                detail="No such dataset"
            )
        files = omcore.File.list(kg_client, file_repository=dataset_obj.repository)
        if len(files) > 100:  # todo: figure out what a reasonable limit is
            raise HTTPException(
                status_code=status_codes.HTTP_400_BAD_REQUEST,
                detail="This dataset has too many files for this query"
            )
        if len(files) > 0:
            filters["inputs"].extend(as_list(files))
        # todo: support FileBundle
    # filter by simulation
    if simulation:
        # todo: add a query for released simulations
        simulation_obj = omcmp.Simulation.from_id(str(simulation), kg_client, release_status="any")
        if simulation_obj is None:
            raise HTTPException(
                status_code=status_codes.HTTP_404_NOT_FOUND,
                detail="No such simulation, or you don't have access."
            )
        filters["inputs"].extend(as_list(simulation_obj.outputs))
    # filter by input_data
    if input_data:
        filters["inputs"].extend(as_list(input_data))
    # filter by software
    if software:
        filters["inputs"].extend(as_list(software))
        environments = omcmp.Environment.list(kg_client, software=software, release_status="any", space=space)
        filters["environment"].extend(as_list(environments))
    # filter by hardware platform
    if platform:
        hardware_obj = omcmp.HardwareSystem.by_name(platform.value, kg_client, release_status="any", space="common")
        # todo: handle different versions of hardware platforms
        environments = omcmp.Environment.list(kg_client, hardware=hardware_obj, release_status="any", space=space)
        filters["environment"].extend(as_list(environments))
    # filter by status
    if status:
        filters["status"] = ACTION_STATUS_TYPES[status.value]
    # filter by tag
    if tags:
        filters["tags"] = tags

    for key in ("inputs", "environment"):
        if key in filters and len(filters[key]) == 0:
            del filters[key]

    visualisation_objects = omcmp.Visualization.list(kg_client, release_status="any", api="query",
                                                    size=size, from_index=from_index,
                                                    space=space)
    return [Visualisation.from_kg_object(obj, kg_client) for obj in visualisation_objects]


@router.post("/visualisations/", response_model=Visualisation, status_code=status_codes.HTTP_201_CREATED)
def create_visualisation(
    visualisation: Visualisation,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a visualisation stage in the Knowledge Graph.
    """
    return create_computation(Visualisation, omcmp.Visualization, visualisation, space, token)


@router.get("/visualisations/{visualisation_id}", response_model=Visualisation)
def get_visualisation(visualisation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific visualisation record, identified by its ID.

    You may only retrieve public records, records in your private space,
    or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        visualisation_object = omcmp.Visualization.from_uuid(str(visualisation_id), kg_client, release_status="any")
    except TypeError as err:
        raise NotFoundError("visualisation", visualisation_id)
    if visualisation_object is None:
        raise NotFoundError("visualisation", visualisation_id)
    return Visualisation.from_kg_object(visualisation_object, kg_client)


@router.put("/visualisations/{visualisation_id}", response_model=Visualisation)
def replace_visualisation(
    visualisation_id: UUID,
    visualisation: Visualisation,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Replace a visualisation record in its entirety.

    You may only replace records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return replace_computation(Visualisation, omcmp.Visualization, visualisation_id, visualisation, token)


@router.patch("/visualisations/{visualisation_id}", response_model=Visualisation)
def update_visualisation(
    visualisation_id: UUID,
    patch: VisualisationPatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a visualisation record.

    You may only update records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(Visualisation, omcmp.Visualization, visualisation_id, patch, token)


@router.delete("/visualisations/{visualisation_id}", response_model=Visualisation)
def delete_visualisation(visualisation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a visualisation record.

    You may only delete records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.Visualization, visualisation_id, token)
