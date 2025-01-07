"""
docstring goes here
"""

"""
   Copyright 2021-2024 CNRS

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

from typing import List, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, Header, Query, HTTPException, status as status_codes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import AnyHttpUrl

from .data_models import Accessibility, EthicsAssessment, License, DOI, DatasetGenerationRequest
from .generator import generate_dataset

logger = logging.getLogger("ebrains-prov-api")

auth = HTTPBearer()
router = APIRouter()


@router.post("/datasets/", status_code=status_codes.HTTP_201_CREATED)
def create_dataset(
    source: DatasetGenerationRequest,
    token: HTTPAuthorizationCredentials = Depends(auth)
):
    """
    Generate a dataset, including a Data Descriptor document, from a list of workflow executions.
    """
    generate_dataset(
        token=token,
        **source,
    )
