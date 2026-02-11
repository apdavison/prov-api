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
import logging


from fastapi import APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import fairgraph.openminds.computation as omcmp

from ..auth.utils import get_kg_client_for_user_account
from .data_models import WorkflowCount


logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.get("/statistics/spaces/", response_model=List[WorkflowCount])
def query_spaces(
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    kg_client = get_kg_client_for_user_account(token.credentials)
    accessible_spaces = kg_client.spaces(permissions=None, names_only=True)
    spaces_with_workflows = ["myspace"]
    spaces_with_workflows.extend(sp for sp in accessible_spaces if sp.startswith("collab"))
    if "computation" in accessible_spaces:
        spaces_with_workflows.append("computation")
    counts = []
    for space in spaces_with_workflows:
        try:
            count = omcmp.WorkflowExecution.count(kg_client, release_status="any", space=space)
        except Exception as err:
            logger.warning(err)
            count = 0
        counts.append(
            WorkflowCount(
                space=space,
                count=count
            )
        )
    return counts
