from distutils.errors import LibError
import re
from typing import List
from enum import Enum
from uuid import UUID
from pydantic import AnyHttpUrl, AnyUrl, BaseModel, Field

from fairgraph.utility import as_list
from fairgraph import IRI
from fairgraph.queries import QueryProperty, Query, Filter
import fairgraph.openminds.core as omcore
import fairgraph.openminds.controlled_terms as omterms
import fairgraph.openminds.computation as omcmp
from ..common.data_models import (
    Person,
    get_repository_host,
    get_repository_iri,
    get_repository_name,
    get_repository_type,
)
from ..common.utils import invert_dict, collab_id_from_space



class WorkflowRecipeType(str, Enum):
    cwl = "CWL workflow"
    cwlcmd = "CWL command line tool"
    snakemake = "Snakemake workflow"
    unicore = "UNICORE workflow"
    generic_python = "Python script"
    jupyter = "Jupyter notebook"


content_type_lookup = {
    "application/vnd.commonworkflowlanguage.workflow": WorkflowRecipeType.cwl,
    "application/vnd.commonworkflowlanguage.cmdline": WorkflowRecipeType.cwlcmd,
    "application/vnd.snakemake.workflowrecipe": WorkflowRecipeType.snakemake,
    "application/vnd.unicore.workflowrecipe": WorkflowRecipeType.unicore,
    "text/x-python": WorkflowRecipeType.generic_python,
    "application/x-ipynb+json": WorkflowRecipeType.jupyter,
}


class WorkflowRecipe(BaseModel):
    id: UUID = None
    alias: str = None
    custodians: List[Person] = None
    description: str = None
    developers: List[Person] = None
    full_documentation: AnyHttpUrl = None
    homepage: AnyHttpUrl = None
    keywords: List[str] = None
    location: AnyUrl = (
        None  # temporarily allow None, but really this should always be present
    )
    name: str = None
    project_id: str
    type: WorkflowRecipeType = None  # temporarily allow None
    version_identifier: str
    version_innovation: str = None

    @classmethod
    def from_kg_object(cls, recipe_version, client):
        recipe = omcmp.WorkflowRecipe.list(
            client,
            scope="any",
            # space=recipe_version.space,
            versions=recipe_version,
        )[0]
        if recipe_version.custodians:
            custodians = [
                Person.from_kg_object(p, client)
                for p in as_list(recipe_version.custodians)
            ]  # todo: could be Organization
        else:
            custodians = [
                Person.from_kg_object(p, client) for p in as_list(recipe.custodians)
            ]
        if recipe_version.developers:
            developers = [
                Person.from_kg_object(p, client)
                for p in as_list(recipe_version.developers)
            ]
        else:
            developers = [
                Person.from_kg_object(p, client) for p in as_list(recipe.developers)
            ]
        if recipe_version.format:
            type_ = content_type_lookup.get(
                recipe_version.format.resolve(client, scope="any").name, None
            )
        else:
            type_ = None
        if recipe_version.homepage:
            homepage = recipe_version.homepage.value
        elif recipe.homepage:
            homepage = recipe.homepage.value
        else:
            homepage = None
        location = None
        if recipe_version.repository:
            repo_obj = recipe_version.repository.resolve(client, scope="any")
            if repo_obj:
                location = str(repo_obj.iri)
        return cls(
            id=recipe_version.uuid,
            name=recipe_version.full_name or recipe.full_name,
            alias=recipe_version.short_name or recipe.short_name,
            custodians=custodians,
            description=recipe_version.description or recipe.description,
            developers=developers,
            type=type_,
            full_documentation=recipe_version.full_documentation,
            homepage=homepage,
            # keywords=as_list(recipe_version.keywords),  # todo: resolve keyword objects
            location=location,
            version_identifier=recipe_version.version_identifier,
            version_innovation=recipe_version.version_innovation,
            project_id=collab_id_from_space(recipe_version.space)
        )

    def to_kg_object(self, client):
        content_type_name = invert_dict(content_type_lookup).get(self.type, None)
        if content_type_name:
            format = omterms.ContentType.by_name(content_type_name, client)
        else:
            format = None
        return omcmp.WorkflowRecipeVersion(
            full_name=self.name,
            short_name=self.alias,
            # accessibility',
            # copyright',
            custodians=[p.to_kg_object(client) for p in self.custodians],
            description=self.description,
            developers=[p.to_kg_object(client) for p in self.developers],
            # digital_identifier',
            format=format,
            full_documentation=self.full_documentation,
            # funding',
            # has_components',
            homepage=IRI(str(self.homepage)),
            # how_to_cite',
            # is_alternative_version_of',
            # is_new_version_of',
            keywords=self.keywords,
            # licenses',
            # other_contributions',
            # related_publications',
            # release_date',
            repository=omcore.FileRepository(
                name=get_repository_name(self.location),
                iri=get_repository_iri(self.location),
                hosted_by=get_repository_host(self.location),
                type=get_repository_type(self.location),
            ),
            # support_channels',
            version_identifier=self.version_identifier,
            version_innovation=self.version_innovation,
        )


class WorkflowRecipePatch(BaseModel):
    name: str = None
    alias: str = None
    custodians: List[Person] = None
    description: str = None
    developers: List[Person] = None
    type: WorkflowRecipeType = (
        None  # temporarily allow none, until new content types added
    )
    full_documentation: AnyHttpUrl = None
    homepage: AnyHttpUrl = None
    keywords: List[str] = None
    location: AnyUrl
    version_identifier: str
    version_innovation: str = None
