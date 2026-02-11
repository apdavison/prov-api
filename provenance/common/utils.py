from decimal import Decimal
from uuid import uuid4
import itertools
from typing import Union

from fastapi import HTTPException, status

import fairgraph.openminds.computation as omcmp
import fairgraph.errors
from fairgraph.utility import as_list

from ..auth.utils import get_kg_client_for_user_account, is_collab_admin



def create_computation(pydantic_cls, fairgraph_cls, pydantic_obj, space, token, background_tasks=None):
    kg_client = get_kg_client_for_user_account(token.credentials)
    if pydantic_obj.id is not None:
        try:
            kg_computation_object = fairgraph_cls.from_uuid(str(pydantic_obj.id), kg_client, release_status="any")
        except fairgraph.errors.AuthenticationError:
            raise AuthenticationError()
        if kg_computation_object is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A computation with id {pydantic_obj.id} already exists. "
                        "The POST endpoint cannot be used to modify an existing computation record.",
            )
    pydantic_obj.id = uuid4()
    kg_computation_object = pydantic_obj.to_kg_object(kg_client)
    if background_tasks:
            background_tasks.add_task(kg_computation_object.save, kg_client, space=space, recursive=True)
    else:
        try:
            kg_computation_object.save(kg_client, space=space, recursive=True, ignore_duplicates=True)
        except fairgraph.errors.AuthenticationError:
            raise AuthenticationError()
    if background_tasks:
        return pydantic_obj
    else:
        return pydantic_cls.from_kg_object(kg_computation_object, kg_client)


def replace_computation(pydantic_cls, fairgraph_cls, computation_id, pydantic_obj, token):
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        kg_computation_object = fairgraph_cls.from_uuid(str(computation_id), kg_client, release_status="any")
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()

    if not (kg_computation_object.space == "myspace" or is_collab_admin(kg_computation_object.space, token.credentials)):
        raise HTTPException(
            status_code=403,
            detail="You can only replace provenance records in your private space "
                   "or in collab spaces for which you are an administrator."
        )
    if pydantic_obj.id is not None and pydantic_obj.id != computation_id:
        raise HTTPException(
            status_code=400,
            detail="The ID of the payload does not match the URL"
        )
    kg_computation_obj_new = pydantic_obj.to_kg_object(kg_client)
    kg_computation_obj_new.id = kg_computation_object.id
    kg_computation_obj_new.save(kg_client, space=kg_computation_object.space, recursive=True, replace=True)
    return pydantic_cls.from_kg_object(kg_computation_obj_new, kg_client)


def patch_computation(pydantic_cls, fairgraph_cls, computation_id, patch, token):
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        kg_computation_object = fairgraph_cls.from_uuid(str(computation_id), kg_client, release_status="any")
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()

    if not (kg_computation_object.space == "myspace" or is_collab_admin(kg_computation_object.space, token.credentials)):
        raise HTTPException(
            status_code=403,
            detail="You can only modify provenance records in your private space "
                   "or in collab spaces for which you are an administrator."
        )
    if patch.id is not None and patch.id != computation_id:
        raise HTTPException(
            status_code=400,
            detail="Modifying the record ID is not permitted."
        )
    kg_computation_obj_updated = patch.apply_to_kg_object(kg_computation_object, kg_client)
    kg_computation_obj_updated.save(kg_client, space=kg_computation_object.space, recursive=True)
    return pydantic_cls.from_kg_object(kg_computation_obj_updated, kg_client)


def delete_computation(fairgraph_cls, computation_id, token):
    kg_client = get_kg_client_for_user_account(token.credentials)
    try:
        kg_computation_object = fairgraph_cls.from_uuid(str(computation_id), kg_client, release_status="any")
    except fairgraph.errors.AuthenticationError:
        raise AuthenticationError()

    if not (kg_computation_object.space == "myspace" or is_collab_admin(kg_computation_object.space, token.credentials)):
        raise HTTPException(
            status_code=403,
            detail="You can only delete provenance records in your private space "
                   "or in collab spaces for which you are an administrator."
        )
    kg_computation_object.delete(kg_client)


def invert_dict(D):
    return {value: key for key, value in D.items()}


def NotFoundError(computation_type, identifier):
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"We could not find a record of a {computation_type} with identifier {identifier}"
    )

def AuthenticationError():
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized request. Have you supplied a valid token?"
    )


def expand_combinations(D):
    if D:
        keys, values = zip(*D.items())
        return [dict(zip(keys, v)) for v in itertools.product(*[as_list(v) for v in values])]
    else:
        return [D]


def collab_id_from_space(space):
    if space.startswith("collab-"):
        return space[7:]
    else:
        return space


def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
    """Encodes a Decimal as int if there is no exponent, otherwise float."""
    # Code taken from Pydantic v1
    exponent = dec_value.as_tuple().exponent
    if isinstance(exponent, int) and exponent >= 0:
        return int(dec_value)
    else:
        return float(dec_value)