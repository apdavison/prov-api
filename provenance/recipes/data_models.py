from typing import List, Optional
from enum import Enum
from uuid import UUID
from pydantic import AnyHttpUrl, AnyUrl, BaseModel

from fairgraph.utility import as_list
from fairgraph import IRI
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
    id: Optional[UUID] = None
    alias: Optional[str] = None
    custodians: Optional[List[Person]] = None
    description: Optional[str] = None
    developers: Optional[List[Person]] = None
    full_documentation: Optional[AnyHttpUrl] = None
    homepage: Optional[AnyHttpUrl] = None
    keywords: Optional[List[str]] = None
                         # temporarily allow None, but really location should always be present
    location: Optional[AnyUrl] = None
    name: Optional[str] = None
    project_id: str
    type: Optional[WorkflowRecipeType] = None  # temporarily allow None
    version_identifier: str
    version_innovation: Optional[str] = None

    @classmethod
    def from_kg_object(cls, recipe_version, client):
        parents = omcmp.WorkflowRecipe.list(
            client,
            release_status="any",
            # space=recipe_version.space,
            versions=recipe_version,
        )
        if len(parents) == 0:
            return None
        else:
            recipe = parents[0]

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
                recipe_version.format.resolve(client, release_status="any").name, None
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
            repo_obj = recipe_version.repository.resolve(client, release_status="any")
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
        location = str(self.location)
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
                name=get_repository_name(location),
                iri=get_repository_iri(location),
                hosted_by=get_repository_host(location),
                type=get_repository_type(location),
            ),
            # support_channels',
            version_identifier=self.version_identifier,
            version_innovation=self.version_innovation,
        )


class WorkflowRecipePatch(BaseModel):
    name: Optional[str] = None
    alias: Optional[str] = None
    custodians: Optional[List[Person]] = None
    description: Optional[str] = None
    developers: Optional[List[Person]] = None
    # temporarily allow none, until new content types added
    type: Optional[WorkflowRecipeType] = None
    full_documentation: Optional[AnyHttpUrl] = None
    homepage: Optional[AnyHttpUrl] = None
    keywords: Optional[List[str]] = None
    location: AnyUrl
    version_identifier: str
    version_innovation: Optional[str] = None
