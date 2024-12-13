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

from typing import List
from uuid import UUID
from datetime import datetime
import logging

import fairgraph.openminds.computation as omcmp
import fairgraph.errors

from fastapi import APIRouter, Depends, Header, Query, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from ..auth.utils import get_kg_client_for_user_account
from ..common.utils import create_computation, delete_computation, NotFoundError, AuthenticationError
from .data_models import WorkflowExecution
from .. import settings


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/workflows/", response_model=List[WorkflowExecution])
def query_workflows(
    space: str = Query(None, description="Knowledge Graph space to search in"),
    recipe_id: UUID = Query(None, description="Return runs of the workflow recipe with the given ID"),
    tags: List[str] = Query(None, description="Return workflows with _all_ of these tags"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Query recorded workflows, filtered according to various criteria.

    The list may contain records of workflows that are public, were launched by the logged-in user,
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    filters = {}
    if recipe_id:
        filters["recipe"] = recipe_id
    # todo: handle tags
    try:
        workflows = omcmp.WorkflowExecution.list(
            kg_client, scope="any", space=space,
            from_index=from_index, size=size, **filters)
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()

    return [WorkflowExecution.from_kg_object(wf, kg_client) for wf in workflows]


@router.post("/workflows/", response_model=WorkflowExecution, status_code=status.HTTP_201_CREATED)
def store_recorded_workflow(
    workflow: WorkflowExecution,
    background_tasks: BackgroundTasks,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new record of a workflow execution in the Knowledge Graph.
    """
    return create_computation(WorkflowExecution, omcmp.WorkflowExecution, workflow, space, token, background_tasks)


@router.get("/workflows/{workflow_id}", response_model=WorkflowExecution)
def get_recorded_workflow(workflow_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a specific record of a workflow execution from the Knowledge Graph, identified by its ID.

    You may only retrieve public records, records that you created, or records associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        workflow_object = omcmp.WorkflowExecution.from_uuid(str(workflow_id), kg_client, scope="any")
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()
    except TypeError as err:
        raise NotFoundError("workflow execution", workflow_id)
    if workflow_object is None:
        raise NotFoundError("workflow execution", workflow_id)
    return WorkflowExecution.from_kg_object(workflow_object, kg_client)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a record of a computational workflow execution.

    You may only delete records that you created, or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.WorkflowExecution, workflow_id, token)
