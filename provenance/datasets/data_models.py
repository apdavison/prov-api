"""

"""

from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel


class Accessibility(str, Enum):
    # todo: generate from openMINDS ProductAccessibility instances
    free = "free access"
    underembargo = "under embargo"
    restricted = "restricted access"


class EthicsAssessment(str, Enum):
    # todo: generate from openMINDS EthicsAssessment instances
    not_required = "not required"
    sensitive = "EU compliant, sensitive"


class License(str, Enum):
    # todo: generate from openMINDS License instances
    cc0 = "CC0"
    ccby = "CC-BY"


class DOI:
    # todo: add validator rule
    pass


class DatasetGenerationRequest(BaseModel):
    from_workflows: List[UUID]  # = Field(title=, description=)
    name: str
    description: str
    accessibility: Accessibility
    ethics_assessment: EthicsAssessment
    license: License
    funding: Optional[str] = None
    homepage: Optional[AnyHttpUrl] = None
    how_to_cite: Optional[str] = None
    keywords: Optional[
        List[str]
    ] = None  # todo: define an Enum generated from openMINDS instances
    #related_publications=Optional[List[DOI]]
