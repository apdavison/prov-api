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
from fairgraph import KGObject
import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp

from ..auth.utils import get_kg_client_for_user_account

from .data_models import DataCopy, DataCopyPatch
from ..common.data_models import HardwareSystem, Status, ACTION_STATUS_TYPES
from ..common.utils import create_computation, replace_computation, patch_computation, delete_computation, NotFoundError
from .. import settings


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/datacopies/", response_model=List[DataCopy])
def query_data_copies(
    research_product: UUID = Query(None, description="Return records of data copies from this research product"),
    input_data: UUID = Query(None, description="Return records of copies of a given data file or directory containing data files"),
    space: str = Query(None, description="Knowledge Graph space to search in"),
    status: Status = Query(None, description="Return records of data copies with this status"),
    tags: List[str] = Query(None, description="Return records of data copies with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),

):
    """
    docstring goes here
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    filters = {
        "inputs": []
    }
    if research_product:
        rp_obj = KGObject.from_id(research_product, kg_client)
        if rp_obj is None:
            raise HTTPException(
                status_code=status_codes.HTTP_404_NOT_FOUND,
                detail="No such research product"
            )
        files = omcore.File.list(kg_client, file_repository=rp_obj.repository)
        if len(files) > 100:  # todo: figure out what a reasonable limit is
            raise HTTPException(
                status_code=status_codes.HTTP_400_BAD_REQUEST,
                detail="This dataset has too many files for this query"
            )
        if len(files) > 0:
            filters["inputs"].extend(as_list(files))
        # todo: support FileBundle
    # filter by input_data
    if input_data:
        filters["inputs"].extend(as_list(input_data))
    # filter by status
    if status:
        filters["status"] = ACTION_STATUS_TYPES[status.value]
    # filter by tag
    if tags:
        filters["tags"] = tags

    for key in ("inputs",):
        if key in filters and len(filters[key]) == 0:
            del filters[key]

    data_copy_objects = omcmp.DataCopy.list(kg_client, scope="any", api="query",
                                            size=size, from_index=from_index,
                                            space=space)
    return [DataCopy.from_kg_object(obj, kg_client) for obj in data_copy_objects]


@router.post("/datacopies/", response_model=DataCopy, status_code=status_codes.HTTP_201_CREATED)
def create_data_copy(
    data_copy: DataCopy,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a data_copy stage in the Knowledge Graph.
    """
    return create_computation(DataCopy, omcmp.DataCopy, data_copy, space, token)


@router.get("/datacopies/{data_copy_id}", response_model=DataCopy)
def get_data_copy(data_copy_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific data_copy record, identified by its ID.

    You may only retrieve public records, records in your private space,
    or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        data_copy_object = omcmp.DataCopy.from_uuid(str(data_copy_id), kg_client, scope="any")
    except TypeError as err:
        raise NotFoundError("data_copy", data_copy_id)
    if data_copy_object is None:
        raise NotFoundError("data_copy", data_copy_id)
    return DataCopy.from_kg_object(data_copy_object, kg_client)


@router.put("/datacopies/{data_copy_id}", response_model=DataCopy)
def replace_data_copy(
    data_copy_id: UUID,
    data_copy: DataCopy,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Replace a data_copy record in its entirety.

    You may only replace records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return replace_computation(DataCopy, omcmp.DataCopy, data_copy_id, data_copy, token)


@router.patch("/datacopies/{data_copy_id}", response_model=DataCopy)
def update_data_copy(
    data_copy_id: UUID,
    patch: DataCopyPatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a data_copy record.

    You may only update records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(DataCopy, omcmp.DataCopy, data_copy_id, patch, token)


@router.delete("/datacopies/{data_copy_id}", response_model=DataCopy)
def delete_data_copy(data_copy_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a data_copy record.

    You may only delete records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.DataCopy, data_copy_id, token)
