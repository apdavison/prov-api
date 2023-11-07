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

from .data_models import GenericComputation, GenericComputationPatch
from ..common.data_models import HardwareSystem, Status, ACTION_STATUS_TYPES
from ..common.utils import create_computation, replace_computation, patch_computation, delete_computation, NotFoundError
from .. import settings


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/miscellaneous/", response_model=List[GenericComputation])
def query_miscellaneous(
    input_data: UUID = Query(None, description="Return computations using a given data file or directory containing data files"),
    software: UUID = Query(None, description="Return computations that used a specific software version"),
    platform: HardwareSystem = Query(None, description="Return computations that ran on this hardware platform"),
    space: str = Query(None, description="Knowledge Graph space to search in"),
    status: Status = Query(None, description="Return computations with this status"),
    tags: List[str] = Query(None, description="Return computations with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),

):
    """
    Query recorded miscellaneous computations, filtered according to various criteria.
    By "miscellaneous" we means that the computations do not fit into the EBRAINS classification:
    ("data analysis", "simulation", "visualisation", "optimisation", "data copy")

    Where multiple filters are applied, they are combined with AND,
    e.g. software=<UUID for Elephant v0.9.0>&platform=pizdaint returns
    computations that used Elephant v0.9.0 and that ran on Piz Daint.

    The list may contain records of computations that are public, were performed by the logged-in user,
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    # filters = {
    #     "inputs": [],
    #     "environment": []
    # }
    # if dataset:
    #     dataset_obj = omcore.DatasetVersion.from_id(str(dataset), kg_client)
    #     if dataset_obj is None:
    #         raise HTTPException(
    #             status_code=status_codes.HTTP_404_NOT_FOUND,
    #             detail="No such dataset"
    #         )
    #     files = omcore.File.list(kg_client, file_repository=dataset_obj.repository)
    #     if len(files) > 100:  # todo: figure out what a reasonable limit is
    #         raise HTTPException(
    #             status_code=status_codes.HTTP_400_BAD_REQUEST,
    #             detail="This dataset has too many files for this query"
    #         )
    #     if len(files) > 0:
    #         filters["inputs"].extend(as_list(files))
    #     # todo: support FileBundle
    # # filter by simulation
    # if simulation:
    #     # todo: add a query for released simulations
    #     simulation_obj = omcmp.Simulation.from_id(str(simulation), kg_client, scope="any")
    #     if simulation_obj is None:
    #         raise HTTPException(
    #             status_code=status_codes.HTTP_404_NOT_FOUND,
    #             detail="No such simulation, or you don't have access."
    #         )
    #     filters["inputs"].extend(as_list(simulation_obj.outputs))
    # # filter by input_data
    # if input_data:
    #     filters["inputs"].extend(as_list(input_data))
    # # filter by software
    # if software:
    #     filters["inputs"].extend(as_list(software))
    #     environments = omcmp.Environment.list(kg_client, software=software, scope="any", space=space)
    #     filters["environment"].extend(as_list(environments))
    # # filter by hardware platform
    # if platform:
    #     hardware_obj = omcmp.HardwareSystem.by_name(platform.value, kg_client, scope="any", space="common")
    #     # todo: handle different versions of hardware platforms
    #     environments = omcmp.Environment.list(kg_client, hardware=hardware_obj, scope="any", space=space)
    #     filters["environment"].extend(as_list(environments))
    # # filter by status
    # if status:
    #     filters["status"] = ACTION_STATUS_TYPES[status.value]
    # # filter by tag
    # if tags:
    #     filters["tags"] = tags

    # for key in ("inputs", "environment"):
    #     if key in filters and len(filters[key]) == 0:
    #         del filters[key]

    # computation_objects = omcmp.GenericComputation.list(kg_client, scope="any", api="query",
    #                                                 size=size, from_index=from_index,
    #                                                 space=space)
    # return [obj.from_kg_object(kg_client) for obj in computation_objects]


@router.post("/miscellaneous/", response_model=GenericComputation, status_code=status_codes.HTTP_201_CREATED)
def create_computation(
    computation: GenericComputation,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a miscellaneous computation stage in the Knowledge Graph.
    """
    return create_computation(GenericComputation, omcmp.GenericComputation, computation, space, token)


@router.get("/miscellaneous/{computation_id}", response_model=GenericComputation)
def get_computation(computation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific miscellaneous computation record, identified by its ID.

    You may only retrieve public records, records in your private space,
    or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        computation_object = omcmp.GenericComputation.from_uuid(str(computation_id), kg_client, scope="any")
    except TypeError as err:
        raise NotFoundError("computation", computation_id)
    if computation_object is None:
        raise NotFoundError("computation", computation_id)
    return GenericComputation.from_kg_object(computation_object, kg_client)


@router.put("/miscellaneous/{computation_id}", response_model=GenericComputation)
def replace_computation(
    computation_id: UUID,
    computation: GenericComputation,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Replace a miscellaneous computation record in its entirety.

    You may only replace records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return replace_computation(GenericComputation, omcmp.GenericComputation, computation_id, computation, token)


@router.patch("/miscellaneous/{computation_id}", response_model=GenericComputation)
def update_computation(
    computation_id: UUID,
    patch: GenericComputationPatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a miscellaneous computation record.

    You may only update records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(GenericComputation, omcmp.GenericComputation, computation_id, patch, token)


@router.delete("/analyses/{computation_id}", response_model=GenericComputation)
def delete_computation(computation_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a miscellaneous computation record.

    You may only delete records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.GenericComputation, computation_id, token)
