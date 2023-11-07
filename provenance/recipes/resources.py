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
from uuid import UUID, uuid4
import logging

from fairgraph.utility import as_list
import fairgraph.openminds.computation as omcmp
import fairgraph.errors

from fastapi import APIRouter, Depends, Query, HTTPException, status as status_codes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..auth.utils import get_kg_client_for_user_account
from ..common.utils import patch_computation, delete_computation, NotFoundError, AuthenticationError
from .data_models import WorkflowRecipe, WorkflowRecipePatch


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/recipes/", response_model=List[WorkflowRecipe])
def query_workflow_recipes(
    space: str = Query(None, description="Knowledge Graph space to search in"),
    size: int = Query(100, description="Number of records to return"),
    from_index: int = Query(0, description="Index of the first record to return"),
    # from header
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Query workflow recipes.

    The list may contain records of recipes that are public
    or that are associated with a collab of which the user is a member.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        recipes = omcmp.WorkflowRecipeVersion.list(
            kg_client, scope="any", space=space, api="core",
            from_index=from_index, size=size)
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()
    return [WorkflowRecipe.from_kg_object(rcp, kg_client) for rcp in recipes]


@router.get("/recipes/{recipe_id}", response_model=WorkflowRecipe)
def get_workflow_recipe(recipe_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Retrieve a workflow recipe (aka workflow description) from the Knowledge Graph, identified by its ID.

    You may only retrieve public recipes, recipes that you created, or recipes associated with a collab which you can view.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        recipe_object = omcmp.WorkflowRecipeVersion.from_uuid(str(recipe_id), kg_client, scope="any")
    except TypeError as err:
        raise NotFoundError("workflow recipe", recipe_id)
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()
    if recipe_object is None:
        raise NotFoundError("workflow recipe", recipe_id)
    return WorkflowRecipe.from_kg_object(recipe_object, kg_client)


@router.post("/recipes/", response_model=WorkflowRecipe, status_code=status_codes.HTTP_201_CREATED)
def create_workflow_recipe(
    recipe: WorkflowRecipe,
    space: str = "myspace",
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Store a new recipe, or a new version of an existing recipe, in the Knowledge Graph.
    """
    kg_client = get_kg_client_for_user_account(token.credentials)
    requested_recipe_uuid = None
    if recipe.id is not None:
        kg_recipe_version = omcmp.WorkflowRecipeVersion.from_uuid(str(recipe.id), kg_client, scope="any")
        if kg_recipe_version is not None:
            raise HTTPException(
                status_code=status_codes.HTTP_400_BAD_REQUEST,
                detail=f"A workflow recipe version with id {recipe.id} already exists. "
                        "The POST endpoint cannot be used to modify an existing version of a workflow recipe.",
            )
        # we remove the requested id for now, so we can check no recipe with the same name and version already exists
        requested_recipe_uuid = str(recipe.id)
        recipe.id = None
    kg_recipe_version = recipe.to_kg_object(kg_client)

    if kg_recipe_version.exists(kg_client):
        raise HTTPException(
            status_code=status_codes.HTTP_400_BAD_REQUEST,
            detail=f"A workflow recipe version with name '{kg_recipe_version.name}' "
                   f"and version {kg_recipe_version.version_identifier} already exists. "
                    "The POST endpoint cannot be used to modify an existing version of a workflow recipe.",
        )

    # now we replace the requested recipe id, if any
    if requested_recipe_uuid:
        kg_recipe_version.id = kg_client.uri_from_uuid(requested_recipe_uuid)

    # try to figure out if this is a new version of an existing recipe
    # todo in future: also search released workflow recipe
    alternative_versions = None
    parent_workflow = omcmp.WorkflowRecipe.list(kg_client, space=space, scope="any",
                                                name=recipe.name)
    if parent_workflow:
        if len(parent_workflow) > 1:
            # KG query returns not-exact matches, need to filter further
            candidate_parents = [pwf for pwf in parent_workflow if pwf.name == recipe.name]
            if len(candidate_parents) == 1:
                parent_workflow = candidate_parents[0]
            else:
                parent_workflow = None
        elif parent_workflow[0].name == recipe.name:
            parent_workflow = parent_workflow[0]
        else:
            parent_workflow = None
    if parent_workflow:
        parent_workflow.resolve(kg_client, scope="any", follow_links=1)
        if parent_workflow.versions:
            alternative_versions = sorted(
                as_list(parent_workflow.versions),
                key=lambda wfv: wfv.version_identifier
            )
            kg_recipe_version.is_new_version_of = alternative_versions[-1]
        parent_workflow.versions = as_list(parent_workflow.versions) + [kg_recipe_version]
    else:
        parent_workflow = omcmp.WorkflowRecipe(
            name=kg_recipe_version.name,
            alias=kg_recipe_version.alias,
            description=kg_recipe_version.description,
            developers=kg_recipe_version.developers,
            versions=[kg_recipe_version])
    kg_recipe_version.save(kg_client, space=space, recursive=True)
    parent_workflow.save(kg_client, space=parent_workflow.space or space, recursive=False)
    return WorkflowRecipe.from_kg_object(kg_recipe_version, kg_client)


@router.patch("/recipes/{recipe_id}", response_model=WorkflowRecipe)
def update_workflow_recipe(
    recipe_id: UUID,
    patch: WorkflowRecipePatch,
    token: HTTPAuthorizationCredentials = Depends(auth),
):
    """
    Modify part of the metadata in a workflow recipe.

    You may only update records in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return patch_computation(WorkflowRecipe, omcmp.WorkflowRecipeVersion, recipe_id, patch, token)


@router.delete("/recipes/{recipe_id}")
def delete_workflow_recipe(recipe_id: UUID, token: HTTPAuthorizationCredentials = Depends(auth)):
    """
    Delete a workflow recipe.

    You may only delete recipes in your private space,
    or that are associated with a collab of which you are an administrator.
    """
    return delete_computation(omcmp.WorkflowRecipeVersion, recipe_id, token)
