"""


"""

import logging
import json
import requests
from fastapi import HTTPException, status

from fairgraph.client import KGClient
from fairgraph.errors import AuthenticationError

from .. import settings
from . import oauth

logger = logging.getLogger("ebrains-prov-api")

kg_client_for_service_account = None


def get_kg_client_for_service_account():
    global kg_client_for_service_account
    if kg_client_for_service_account is None:
        kg_client_for_service_account = KGClient(
            client_id=settings.KG_SERVICE_ACCOUNT_CLIENT_ID,
            client_secret=settings.KG_SERVICE_ACCOUNT_SECRET,
            host=settings.KG_CORE_API_HOST,
        )
    return kg_client_for_service_account


def get_kg_client_for_user_account(token):
    try:
        return KGClient(token=token, host=settings.KG_CORE_API_HOST)
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


async def can_read_space(space, token):
    user_client = get_kg_client_for_user_account(token=token)
    return space in user_client.spaces(permissions=["read"], names_only=True)


async def get_user_from_token(user_token):
    """
    Get user id with token
    :param request: request
    :type request: str
    :returns: res._content
    :rtype: str
    """
    # collab v2 only
    user_info = await oauth.ebrains.user_info(
        token={"access_token": user_token, "token_type": "bearer"}
    )
    if "error" in user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=user_info["error_description"]
        )
    user_info["id"] = user_info["sub"]
    user_info["username"] = user_info.get("preferred_username", "unknown")
    return user_info


async def get_collab_info(collab_id, user_token):
    collab_info_url = f"{settings.HBP_COLLAB_SERVICE_URL_V2}collabs/{collab_id}"
    headers = {"Authorization": f"Bearer {user_token}"}
    res = requests.get(collab_info_url, headers=headers)
    try:
        response = res.json()
    except json.decoder.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid collab id"
        )
    return response


async def get_collab_permissions(collab_id, user_token):
    if collab_id.startswith("collab-"):
        collab_id = collab_id[7:]
    user_info = get_user_from_token(user_token)
    target_team_names = {role: f"collab-{collab_id}-{role}"
                         for role in ("viewer", "editor", "administrator")}

    highest_collab_role = None
    for role, team_name in target_team_names.items():
        if team_name in user_info["roles"]["team"]:
            highest_collab_role = role
    if highest_collab_role == "viewer":
        permissions = {"VIEW": True, "UPDATE": False, "DELETE": False}
    elif highest_collab_role == "editor":
        permissions = {"VIEW": True, "UPDATE": True, "DELETE": False}
    elif highest_collab_role == "administrator":
        permissions = {"VIEW": True, "UPDATE": True, "DELETE": True}
    else:
        assert highest_collab_role is None
        collab_info = await get_collab_info(collab_id, user_token)
        if collab_info.get("isPublic", False):  # will be False if 404 collab not found
            permissions = {"VIEW": True, "UPDATE": False, "DELETE": False}
        else:
            permissions = {"VIEW": False, "UPDATE": False, "DELETE": False}
    return permissions


async def is_collab_member(collab_id, user_token):
    if collab_id is None:
        return False
    try:
        int(collab_id)
    except ValueError:
        permissions = await get_collab_permissions(collab_id, user_token)
        return permissions.get("UPDATE", False)
    else:
        return False


async def is_collab_admin(collab_id, user_token):
    if collab_id is None:
        return False
    try:
        int(collab_id)
    except ValueError:
        permissions = await get_collab_permissions(collab_id, user_token)
        return permissions.get("DELETE", False)
    else:
        return False


async def is_global_admin(user_token):
    user_info = get_user_from_token(user_token)
    return f"group-{settings.ADMIN_GROUP_ID}" in user_info["groups"]


async def can_view_collab(collab_id, user_token):
    if collab_id is None:
        return False
    try:
        int(collab_id)
    except ValueError:
        get_collab_permissions = get_collab_permissions_v2
        permissions = await get_collab_permissions(collab_id, user_token)
        return permissions.get("VIEW", False)
    else:
        return False


async def get_editable_collabs(user_token):
    # collab v2 only
    user_info = get_user_from_token(user_token)
    editable_collab_ids = set()
    for team_name in user_info["roles"]["team"]:
        if team_name.endswith("-editor") or team_name.endswith("-administrator"):
            collab_id = "-".join(team_name.split("-")[1:-1])
            editable_collab_ids.add(collab_id)
    return sorted(editable_collab_ids)
