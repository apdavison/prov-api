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

from enum import Enum
from os import environ
from uuid import UUID
from typing import List, Union, Optional
import re
import hashlib
import json
from fairgraph.registry import lookup_by_id
from fairgraph.utility import as_list

try:
    from typing import Literal  # Python >= 3.8
except ImportError:
    from typing_extensions import Literal
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, AnyUrl, Field

from fairgraph.base_v3 import KGProxy as KGProxy, IRI, as_list
#from fairgraph.openminds import controlledterms
from fairgraph.openminds.core.miscellaneous.quantitative_value import QuantitativeValue
import fairgraph.openminds.core as omcore
import fairgraph.openminds.computation as omcmp
from fairgraph.openminds.controlledterms import FileRepositoryType, UnitOfMeasurement, ActionStatusType

from .examples import EXAMPLES
from ..auth.utils import get_kg_client_for_service_account


status_name_map = {
    "active": "running",
    "completed": "completed",
    "failed": "failed",
    "potential": "queued"
}

def _get_action_status_types():
    kg_client_service_account = get_kg_client_for_service_account()
    return ActionStatusType.list(kg_client_service_account, scope="in progress", 
                                 api="core", space="controlled", size=10)


ACTION_STATUS_TYPES = _get_action_status_types()

Status = Enum(
    "Status",
    [(status_name_map[ast.name], status_name_map[ast.name]) 
     for ast in ACTION_STATUS_TYPES]
)

ACTION_STATUS_TYPES = {status_name_map[ast.name]: ast for ast in ACTION_STATUS_TYPES}


class CryptographicHashFunction(str, Enum):
    """Algorithm used to compute digest of file contents"""

    sha1 = "SHA-1"
    md5 = "MD5"
    todo = "list to be completed"


def get_identifier(iri, prefix):
    """Return a valid Python variable name based on a KG object UUID"""
    return prefix + "_" + iri.split("/")[-1].replace("-", "")


def _get_units_of_measurement():
    # pre-fetch units of measurement
    kg_client_service_account = get_kg_client_for_service_account()
    units_objects = UnitOfMeasurement.list(kg_client_service_account, api="core", scope="in progress", space="controlled")
    # the follow addition is a temporary workaround with a locally-generated id until core-hour is added
    units_objects.append(UnitOfMeasurement(name="core-hour", id="https://kg.ebrains.eu/api/instances/686f4d65-bdc7-4f69-bf32-4c9f09028541"))
    return units_objects

UNITS = _get_units_of_measurement()

Units = Enum(
    "Units",
    [(get_identifier(unit.id, "u"), unit.name) for unit in UNITS]
)

UNITS = {u: unit_obj for u, unit_obj in zip(Units, UNITS)}


def _get_content_types():
    kg_client_service_account = get_kg_client_for_service_account()
    content_types = omcore.ContentType.list(kg_client_service_account, api="core", scope="in progress", space="controlled", size=10000)
    values = [(get_identifier(ct.id, "ct"), ct.name) for ct in content_types]
    return values


ContentType = Enum(
    "ContentType",
    _get_content_types()
)


class ComputationType(str, Enum):
    visualization = "visualization"
    analysis = "data analysis"
    simulation = "simulation"
    optimization = "optimization"
    #preprocessing = "pre-processing"


class Digest(BaseModel):
    """Hash value of the content of a file, used as a simple way to check if the contents have changed"""

    value: str
    algorithm: CryptographicHashFunction


def _get_hosting_organizations():
    kg_client_service_account = get_kg_client_for_service_account()
    hosting_orgs = {
        name: omcore.Organization.list(kg_client_service_account, scope="in progress", space="common", alias=name)[0]
        for name in ("EBRAINS", "GitHub", "Yale", "EBI", "CERN", "CSCS", "CNRS")
    }
    # CSCS = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/e3f16a1a-184e-447d-aced-375c00ec4d41")
    # GitHub = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/8e16b752-a95a-41f9-acc7-7f7e7c950f1d")
    # Yale = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/5093d906-e058-47e9-a9eb-ac56354f79fc")  # create ModelDB as an org?
    # EBRAINS = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/7dfdd91f-3d05-424a-80bd-6d1d5dc11cd3")
    # CERN = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/dbf4d089-9be1-4420-822b-87ecb7204840")  # create Zenodo as an org?
    # EBI = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/30aa86d9-39b0-45d1-a8c3-a76d64bfe57a")  # create BioModels as an org?
    # Bitbucket = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/574d7d5c-056a-4dae-9d1c-921057451199")
    # CNRS = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/31259b06-91d0-4ad8-acfd-303fc9ed613b")
    # JSC = KGProxy(omcore.Organization, "https://kg.ebrains.eu/api/instances/<TO DO>")
    return hosting_orgs


FILE_HOSTS = _get_hosting_organizations()


file_location_patterns = {
    "https://object.cscs.ch": FILE_HOSTS["EBRAINS"],
    "swift://cscs.ch": FILE_HOSTS["EBRAINS"],
    "https://ksproxy.cscs.ch": FILE_HOSTS["EBRAINS"],
    "https://kg.humanbrainproject.org/proxy/export": FILE_HOSTS["EBRAINS"],
    "https://github.com": FILE_HOSTS["GitHub"],
    "https://senselab.med.yale.edu": FILE_HOSTS["Yale"],
    "http://modeldb.yale.edu": FILE_HOSTS["Yale"],
    "http://example.com": None,
    "https://collab.humanbrainproject.eu": FILE_HOSTS["EBRAINS"],
    "collab://": FILE_HOSTS["EBRAINS"],
    "https://drive.ebrains.eu": FILE_HOSTS["EBRAINS"],
    "https://zenodo.org": FILE_HOSTS["CERN"],
    "https://www.ebi.ac.uk": FILE_HOSTS["EBI"],
    #"https://CrimsonWhite@bitbucket.org": FILE_HOSTS["Bitbucket"],
    "http://cns.iaf.cnrs-gif.fr": FILE_HOSTS["CNRS"],
    "https://gpfs-proxy.brainsimulation.eu/cscs": FILE_HOSTS["CSCS"],
    #"https://gpfs-proxy.brainsimulation.eu/jsc": JSC,
}


def get_repository_host(url):
    for fragment, org in file_location_patterns.items():
        if fragment in url:
            return org
    return None


CSCS_pattern = r"https://object\.cscs\.ch/v1/(?P<proj>\w+)/(?P<container_name>[\w\.-]+)/(?P<path>\S*)"
GPFS_proxy_pattern = r"https://gpfs-proxy\.brainsimulation\.eu/(?P<site>\w+)/(?P<project_name>[\w-]+)/(?P<path>\S*)"
EBRAINS_Gitlab_pattern = r"https://gitlab.ebrains.eu/(?P<org>[\w-]+)/(?P<project_name>[/\w-]+)/-/"
EBRAINS_Gitlab_pattern2 = r"https://gitlab.ebrains.eu/(?P<org>[\w-]+)/(?P<project_name>[/\w-]+)"

def get_repository_iri(url):
    templates = (
        (CSCS_pattern, "https://object.cscs.ch/v1/{proj}/{container_name}"),
        (GPFS_proxy_pattern, "https://gpfs-proxy.brainsimulation.eu/{site}/{project_name}"),
        (EBRAINS_Gitlab_pattern, "https://gitlab.ebrains.eu/{org}/{project_name}"),
        (EBRAINS_Gitlab_pattern2, "https://gitlab.ebrains.eu/{org}/{project_name}"),
    )
    for pattern, template in templates:
        match = re.match(pattern, url)
        if match:
            return IRI(template.format(**match.groupdict()))

    if url.startswith("https://drive.ebrains.eu"):
        return IRI("https://drive.ebrains.eu")
    raise NotImplementedError(f"Repository IRI format not yet supported. Value was {url}")


def get_repository_name(url):
    templates = (
        (CSCS_pattern, "container_name"),
        (GPFS_proxy_pattern, "project_name"),
        (EBRAINS_Gitlab_pattern, "project_name"),
        (EBRAINS_Gitlab_pattern2, "project_name")
    )
    for pattern, key in templates:
        match = re.match(pattern, url)
        if match:
            return match[key]

    if url.startswith("https://drive.ebrains.eu"):
        return "EBRAINS Drive"
    raise NotImplementedError(f"Repository IRI format not yet supported. Value was {url}")


def _get_repository_types():
    kg_client_service_account = get_kg_client_for_service_account()
    return {
        obj.name: obj for obj in FileRepositoryType.list(kg_client_service_account, scope="released")
    }

REPOSITORY_TYPES = _get_repository_types()

def get_repository_type(url):
    if url.startswith("https://object.cscs.ch"):
        return REPOSITORY_TYPES["Swift repository"]
    elif url.startswith("https://gpfs-proxy.brainsimulation.eu"):
        return REPOSITORY_TYPES["GPFS repository"]
    elif url.startswith("https://drive.ebrains.eu"):
        return REPOSITORY_TYPES["Seafile repository"]
    elif url.startswith("https://gitlab.ebrains.eu"):
        return REPOSITORY_TYPES["GitLab repository"]
    elif url.startswith("https://github.com"):
        return REPOSITORY_TYPES["GitHub repository"]
    raise NotImplementedError(f"Repository IRI format not yet supported. Value was {url}")


class File(BaseModel):
    """Metadata about a file"""

    description: Optional[str] = Field(None, title="Description of the file contents")
    format: Optional[ContentType] = Field(None, title="Content type of the file, expressed as a media type string")
    hash: Digest = None
    location: AnyUrl = None  # for files generated within workflows but not preserved, location may be None
    file_name: str
    size: Optional[int] = Field(None, title="File size in bytes")
    # bundle
    # repository

    class Config:
        schema_extra = {"example": EXAMPLES["File"]}

    @classmethod
    def from_kg_object(cls, file_object, client):
        if isinstance(file_object, KGProxy):
            file_object = file_object.resolve(client, scope="in progress")
        if file_object.format:
            name = file_object.format.resolve(client, scope="in progress").name
            format = ContentType(name)
        else:
            format = None
        if file_object.hash:
            def get_algorithm(name):
                return CryptographicHashFunction(name)
            if isinstance(file_object.hash, list):
                hash = Digest(value=file_object.hash[0].digest, algorithm=get_algorithm(file_object.hash[0].algorithm))
            else:
                hash = Digest(value=file_object.hash.digest, algorithm=get_algorithm(file_object.hash.algorithm))
        else:
            hash = None
        if file_object.storage_size:
            size = int(file_object.storage_size.value)
        else:
            size = None
        if isinstance(file_object, omcore.File):
            location = file_object.iri.value
        else:
            assert isinstance(file_object, omcmp.LocalFile)
            location = f"file://{file_object.path}"
        return cls(
            format=format,
            hash=hash,
            location=location,
            file_name=file_object.name,
            size=size,
            description=file_object.content_description
        )

    def to_kg_object(self, client):
        if self.location and self.location.startswith("http"):
            file_repository = omcore.FileRepository(
                hosted_by=get_repository_host(self.location),
                iri=get_repository_iri(self.location),
                name=get_repository_name(self.location),
                repository_type=get_repository_type(self.location)
            )
        else:
            file_repository = None
        if self.format:
            content_type = omcore.ContentType(name=self.format.value)
        else:
            # todo: if self.format is empty, we should try to infer it
            content_type = None
        hash = omcore.Hash(algorithm=self.hash.algorithm.value, digest=self.hash.value)
        if self.size is None:
            storage_size = None
        else:
            storage_size = QuantitativeValue(value=float(self.size), unit=UNITS[Units("byte")])
        if file_repository:
            file_obj = omcore.File(
                file_repository=file_repository,
                format=content_type,
                hash=hash,
                iri=IRI(self.location),
                name=self.file_name,
                storage_size=storage_size,
                content_description=self.description
            )
        else:
            if self.location:
                path = self.location.replace("file://", "")
            else:
                path = self.file_name
            file_obj = omcmp.LocalFile(
                path=path,
                format=content_type,
                hash=hash,
                name=self.file_name,
                storage_size=storage_size,
                content_description=self.description
            )
        return file_obj


def _get_hardware_systems():
    kg_client_service_account = get_kg_client_for_service_account()
    return {
        obj.name: obj for obj in omcmp.HardwareSystem.list(kg_client_service_account, scope="in progress", space="common")
    }


HARDWARE_SYSTEMS = _get_hardware_systems()


HardwareSystem = Enum(
    "HardwareSystem",
    [(name.lower().replace(" ", ""), name) for name in HARDWARE_SYSTEMS]
)


class StringParameter(BaseModel):
    """A parameter whose value is a string"""

    name: str
    value: str

    class Config:
        schema_extra = {
            "example": {
                "name": "method",
                "value": "simulated_annealing"
            }
        }

    @classmethod
    def from_kg_object(cls, param):
        return cls(
            name=param.name,
            value=param.value
        )

    def to_kg_object(self, client):
        return omcore.StringParameter(name=self.name, value=self.value)


class NumericalParameter(BaseModel):
    """A parameter whose value is a number or physical quantity"""

    name: str
    value: Decimal
    units: Units = None

    class Config:
        schema_extra = {
            "example": {
                "name": "Rm",
                "value": 100.3,
                "units": "MΩ"
            }
        }

    def __str__(self):
        return f"{self.name} = {self.value} {self.units}"

    @classmethod
    def from_kg_object(cls, param):
        return cls(
            name=param.name,
            value=param.values[0].value,
            units=Units(param.values[0].units)
        )

    def to_kg_object(self, client):
        return omcore.NumericalParameter(
            name=self.name,
            values=QuantitativeValue(value=self.value, units=UNITS[self.units])
        )


class ParameterSet(BaseModel):
    """A collection of parameters"""

    items: List[Union[StringParameter, NumericalParameter]]
    description: str = None

    @property
    def identifier(self):
        return hashlib.sha1(
            json.dumps(
                [str(item) for item in self.items]
            ).encode("utf-8")).hexdigest()

    @classmethod
    def from_kg_object(cls, ps_object, client):
        items = []
        for param in as_list(ps_object.parameters):
            if isinstance(param, omcore.NumericalParameter):
                items.append(NumericalParameter.from_kg_object(param))
            elif isinstance(param, omcore.StringParameter):
                items.append(StringParameter.from_kg_object(param))
            else:
                raise TypeError("unexpected object type in parameter set")
        return cls(
            items=items,
            description=ps_object.context
        )

    def to_kg_object(self, client):
        return omcore.ParameterSet(
            parameters=[item.to_kg_object(client) for item in self.items],
            context=self.description
        )


class Person(BaseModel):
    """A human person responsible for launching a computation"""

    given_name: str
    family_name: str
    orcid: str = None

    class Config:
        schema_extra = {
            "example": {
                "family_name": "Destexhe",
                "given_name": "Alain",
                "orcid": "https://orcid.org/0000-0001-7405-0455"
            }
        }

    @classmethod
    def from_kg_object(cls, person, client):
        person = person.resolve(client, scope="in progress")
        orcid = None
        if person.digital_identifiers:
            for digid in as_list(person.digital_identifiers):
                if isinstance(digid, omcore.ORCID):
                    orcid = digid.identifier
                    break
                elif isinstance(digid, KGProxy) and digid.cls == omcore.ORCID:
                    orcid = digid.resolve(client, scope="in progress").identifier
                    break
        return cls(given_name=person.given_name, family_name=person.family_name,
                   orcid=orcid)

    def to_kg_object(self, client):
        obj = omcore.Person(family_name=self.family_name, given_name=self.given_name)
        if self.orcid:
            obj.digital_identifiers = [omcore.ORCID(identifier=self.orcid)]
        return obj


class ResourceUsage(BaseModel):
    """Measurement of the usage of some resource, such as memory, compute time"""

    value: Decimal
    units: Units

    class Config:
        schema_extra = {
            "example": {
                "value": 1017.3,
                "units": "core-hour"
            }
        }

    @classmethod
    def from_kg_object(cls, resource_usage, client):
        return cls(
            value=resource_usage.value,
            units=Units(resource_usage.unit.resolve(client, scope="in progress").name)
        )

    def to_kg_object(self, client):
        # todo: wrap getting units in a function which will produce a meaningful HTTP error message if the unit doesn't exist
        return QuantitativeValue(value=float(self.value), unit=UNITS[self.units])


class SoftwareVersion(BaseModel):
    """Minimal representation of a specific piece of software"""

    id: UUID = None
    software_name: str
    software_version: str

    class Config:
        schema_extra = {
            "example": {
                "software_name": "NEST",
                "software_version": "2.20.0"
            }
        }

    @classmethod
    def from_kg_object(cls, software_version_object, client):
        omcore.SoftwareVersion.set_strict_mode(False)
        svo = software_version_object.resolve(client, scope="in progress")
        omcore.SoftwareVersion.set_strict_mode(True)
        return cls(
            id=client.uuid_from_uri(svo.id),
            software_name=svo.name,
            software_version=svo.version_identifier
        )

    def to_kg_object(self, client):
        omcore.SoftwareVersion.set_strict_mode(False)
        obj = omcore.SoftwareVersion(name=self.software_name, alias=self.software_name,
                                version_identifier=self.software_version)
        omcore.SoftwareVersion.set_strict_mode(True)
        return obj


class ComputationalEnvironment(BaseModel):
    """The environment within which a computation takes place"""

    id: UUID = None
    name: str = Field(..., description="A name/label for this computing environment")
    hardware: HardwareSystem = Field(..., description="The hardware system on which this environment runs")
    configuration: Optional[List[
        ParameterSet
    ]] = Field(None, description="All important hardware settings defining this environment")
    software: Optional[List[
        SoftwareVersion
    ]] = Field(None, description="All software versions available in this environment. Note that the Analysis/Simulation schemas allow storing a list of software versions actually _used_ in a computation")
    description: Optional[str] = Field(None, description="A description of this computing environment")

    class Config:
        schema_extra = {"example": EXAMPLES["ComputationalEnvironment"]}

    @classmethod
    def from_kg_object(cls, env_object, client):
        env = env_object.resolve(client, scope="in progress")
        if env.hardware:
            hardware = HardwareSystem(env.hardware.resolve(client, scope="in progress").name)
        else:
            hardware = None
        return cls(
            id=client.uuid_from_uri(env.id),
            name=env.name,
            hardware=hardware,
            configuration=[ParameterSet.from_kg_object(obj, client) for obj in as_list(env.configuration)],
            software=[SoftwareVersion.from_kg_object(obj, client) for obj in as_list(env.software)],
            description=env.description
        )

    def to_kg_object(self, client):
        return omcmp.Environment(
            name=self.name,
            hardware=HARDWARE_SYSTEMS[self.hardware.value],
            configuration=[conf.to_kg_object(client) for conf in as_list(self.configuration)],
            software=[sv.to_kg_object(client) for sv in as_list(self.software)],
            description=self.description
        )


class LaunchConfiguration(BaseModel):
    """Metadata describing how a computation was launched"""

    description: Optional[str] = Field(None, description="Description of this launch configuration")
    name: Optional[str] = Field(None, description="Label for this launch configuration")
    executable: str = Field(..., description="Path to the command-line executable")
    arguments: Optional[List[str]] = Field(None, description="Command line arguments as a list of strings")
    environment_variables: Optional[ParameterSet] = (
        Field(None, description="Any environment variables defined by this launch configuration")
    )

    class Config:
        schema_extra = {"example": EXAMPLES["LaunchConfiguration"]}

    @property
    def identifier(self):
        return hashlib.sha1(
            json.dumps(
                {
                    "name": self.name,
                    "executable": self.executable,
                    "arguments": self.arguments,
                    "environment_variables": self.environment_variables.identifier if self.environment_variables else None
                }
            ).encode("utf-8")).hexdigest()

    @classmethod
    def from_kg_object(cls, launch_config_object, client):
        lco = launch_config_object.resolve(client, scope="in progress")
        if lco.environment_variables:
            env = ParameterSet.from_kg_object(lco.environment_variables, client)
        else:
            env = None
        return cls(
            description=lco.description,
            name=lco.name,
            executable=lco.executable,
            arguments=lco.arguments,
            environment_variables=env
        )

    def to_kg_object(self, client):
        if self.environment_variables is None:
            env_vars = None
        else:
            self.environment_variables.description = self.environment_variables.description or "environment variables"
            env_vars = self.environment_variables.to_kg_object(client)
        if self.name is None:
            self.name = f"LaunchConfiguration-{self.identifier}"
        return omcmp.LaunchConfiguration(
            name=self.name,
            description=self.description,
            executable=self.executable,
            arguments=self.arguments,
            environment_variables=env_vars
        )


class Computation(BaseModel):
    """
    Abstract base class, should not appear in documentation
    """

    id: Optional[UUID] = Field(None, description="IDs should be valid UUID v4 identifiers. If an ID is not supplied it will be generated by the Knowledge Graph")
    input: List[Union[File, SoftwareVersion]] = Field(..., description="Inputs to this computation (data files and/or code)")
    output: List[File] = Field(..., description="Files generated by this computation")
    environment: ComputationalEnvironment
    launch_config: LaunchConfiguration
    start_time: datetime
    end_time: datetime = None
    started_by: Optional[Person] = Field(None, description="If this field is left blank it is assumed that the account used to upload the provenance metadata is the same as that used to launch the computation")
    status: Status = None
    resource_usage: List[ResourceUsage] = None
    tags: List[str] = None
    description: str = None


class ComputationPatch(Computation):
    """
    Abstract base class, should not appear in documentation
    """

    input: List[Union[File, SoftwareVersion]] = None
    output: List[File] = None
    environment: ComputationalEnvironment = None
    launch_config: LaunchConfiguration = None
    start_time: datetime = None
    end_time: datetime = None
    started_by: Person = None
    status: Status = None
    resource_usage: List[ResourceUsage] = None
    tags: List[str] = None


class ModelVersionReference(BaseModel):
    """
    Reference to a model version.

    The Model Validation API or one of the KG APIs may be used to obtain further
    information about the model version using its ID.
    """

    model_version_id: UUID

    @classmethod
    def from_kg_object(cls, model_version, client):
        return cls(model_version_id=UUID(model_version.uuid))

    def to_kg_object(self, client):
        return omcore.ModelVersion.from_uuid(str(self.model_version_id), client, scope="in progress")
