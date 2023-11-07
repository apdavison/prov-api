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

from typing import List
from datetime import datetime
import logging
from uuid import UUID


from fastapi import APIRouter, Depends, Header, Query, HTTPException, status as status_codes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp
from fairgraph.utility import as_list

from ..common.data_models import HardwareSystem, Status, ACTION_STATUS_TYPES
from .data_models import Optimisation, OptimisationPatch
from ..common.utils import create_computation, replace_computation, patch_computation, delete_computation, NotFoundError
from ..auth.utils import get_kg_client_for_user_account


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/optimisations/", response_model=List[Optimisation])
def query_optimisations(
    model_version: UUID = Query(None, description="Return optimisations of this model version"),
    software: UUID = Query(None, description="Return optimisations that used a specific software version"),
    platform: HardwareSystem = Query(None, description="Return optimisations that ran on this hardware platform"),
    space: str = Query(None, description="Knowledge Graph space to search in"),
    status: Status = Query(None, description="Return optimisations with this status"),
    tags: List[str] = Query(None, description="Return optimisations with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Query recorded optimisations, filtered according to various criteria.

    Where multiple filters are applied, they are combined with AND,
    e.g. software=<UUID for NEST 3.1>&platform=pizdaint returns
    data optimisations that used NEST v3.1 and that ran on pizdaint.

    The list may contain records of data optimisations that are public, were performed by the logged-in user,
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    filters = {
        "inputs": [],
        "environment": []
    }
    if model_version:
        model_version_obj = omcore.ModelVersion.from_id(str(model_version), kg_client)
        if model_version_obj is None:
            raise HTTPException(
                status_code=status_codes.HTTP_404_NOT_FOUND,
                detail="No such model"
            )
        filters["inputs"].append(model_version_obj)
    # filter by software
    if software:
        filters["inputs"].extend(as_list(software))
        environments = omcmp.Environment.list(kg_client, software=software, scope="any", space=space)
        filters["environment"].extend(as_list(environments))
    # filter by hardware platform
    if platform:
        hardware_obj = omcmp.HardwareSystem.by_name(platform.value, kg_client, scope="any", space="common")
        # todo: handle different versions of hardware platforms
        environments = omcmp.Environment.list(kg_client, hardware=hardware_obj, scope="any", space=space)
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

    optimisation_objects = omcmp.Optimization.list(kg_client, scope="any", api="query",
                                                   size=size, from_index=from_index,
                                                   space=space)
    return [Optimisation.from_kg_object(obj, kg_client) for obj in optimisation_objects]


@router.post("/optimisations/", response_model=Optimisation, status_code=status_codes.HTTP_201_CREATED)
def create_optimisation(
    optimisation: Optimisation,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a optimisation stage in the Knowledge Graph.
    """
    return create_computation(Optimisation, omcmp.Optimization, optimisation, space, token)


@router.get("/optimisations/{optimisation_id}", response_model=Optimisation)
def get_optimisation(optimisation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific optimisation record, identified by its ID.

    You may only retrieve public records, records in your private space,
    or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        optimisation_object = omcmp.Optimization.from_uuid(str(optimisation_id), kg_client, scope="any")
    except TypeError as err:
        raise NotFoundError("optimisation", optimisation_id)
    if optimisation_object is None:
        raise NotFoundError("optimisation", optimisation_id)
    return Optimisation.from_kg_object(optimisation_object, kg_client)


@router.put("/optimisations/{optimisation_id}", response_model=Optimisation)
def replace_optimisation(
    optimisation_id: UUID,
    optimisation: Optimisation,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Replace a optimisation record in its entirety.

    You may only replace records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return replace_computation(Optimisation, omcmp.Optimization, optimisation_id, optimisation, token)


@router.patch("/optimisations/{optimisation_id}", response_model=Optimisation)
def update_optimisation(
    optimisation_id: UUID,
    patch: OptimisationPatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a optimisation record.

    You may only update records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(Optimisation, omcmp.Optimization, optimisation_id, patch, token)


@router.delete("/analyses/{optimisation_id}", response_model=Optimisation)
def delete_optimisation(optimisation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a optimisation record.

    You may only delete records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.Optimization, optimisation_id, token)
