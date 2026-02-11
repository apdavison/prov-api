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

from typing import List
from uuid import UUID
from datetime import datetime
import logging

from fastapi import APIRouter, Depends, Header, Query, HTTPException, status as status_codes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from fairgraph.utility import as_list
import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp

from ..auth.utils import get_kg_client_for_user_account
from .data_models import Simulation, SimulationPatch, Simulator
from ..common.data_models import HardwareSystem, Status, ACTION_STATUS_TYPES
from ..common.utils import create_computation, replace_computation, patch_computation, delete_computation, NotFoundError
from .. import settings


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/simulations/", response_model=List[Simulation])
def query_simulations(
    model_version: UUID = Query(None, description="Return only simulations of this model version"),
    simulator: Simulator = Query(None, description="Return simulations using this simulator"),
    platform: HardwareSystem = Query(None, description="Return simulations that ran on this hardware platform"),
    space: str = Query(None, description="Knowledge Graph space to search in"),
    status: Status = Query(None, description="Return simulations with this status"),
    tags: List[str] = Query(None, description="Return simulations with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Query recorded simulations, filtered according to various criteria.

    Where multiple filters are applied, they are combined with AND,
    e.g. simulator=nest&platform=pizdaint returns NEST simulations that ran on Piz Daint.

    The list may contain records of simulations that are public, were performed by the logged-in user,
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    filters = {
        "inputs": [],
        "environment": []
    }
    # filter by model version
    if model_version:
        # todo: add a query for un-released model versions
        model_version_obj = omcore.ModelVersion.from_id(str(model_version), kg_client, release_status="released")
        if model_version_obj is None:
            raise HTTPException(
                status_code=status_codes.HTTP_404_NOT_FOUND,
                detail="No such model version, or you don't have access."
            )
        filters["inputs"].append(model_version_obj)
    # filter by simulator
    if simulator:
        filters["inputs"].append(simulator)
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

    simulation_objects = omcmp.Simulation.list(kg_client, release_status="any", api="query",
                                               size=size, from_index=from_index,
                                               space=space)
    return [Simulation.from_kg_object(obj, kg_client) for obj in simulation_objects]


@router.post("/simulations/", response_model=Simulation, status_code=status_codes.HTTP_201_CREATED)
def create_simulation(
    simulation: Simulation,
    space: str = Query(None, description="Knowledge Graph space to save to"),
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a numerical simulation in the Knowledge Graph.
    """
    return create_computation(Simulation, omcmp.Simulation, simulation, space, token)


@router.get("/simulations/{simulation_id}", response_model=Simulation)
def get_simulation(simulation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific simulation record, identified by its ID.

    You may only retrieve public records, records in your private space,
    or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        simulation_object = omcmp.Visualization.from_uuid(str(simulation_id), kg_client, release_status="any")
    except TypeError as err:
        raise NotFoundError("simulation", simulation_id)
    if simulation_object is None:
        raise NotFoundError("simulation", simulation_id)
    return Simulation.from_kg_object(simulation_object, kg_client)


@router.put("/simulations/{simulation_id}", response_model=Simulation)
def replace_simulation(
    simulation_id: UUID,
    simulation: Simulation,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Replace a simulation record in its entirety.

    You may only replace records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return replace_computation(Simulation, omcmp.Simulation, simulation_id, simulation, token)


@router.patch("/simulations/{simulation_id}", response_model=Simulation)
def update_simulation(
    simulation_id: UUID,
    patch: SimulationPatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a simulation record.

    You may only modify records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(Simulation, omcmp.Simulation, simulation_id, patch, token)


@router.delete("/simulations/{simulation_id}")
def delete_simulation(simulation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a simulation record.

    You may only delete records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.Simulation, simulation_id, token)
