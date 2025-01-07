"""
This file is for the development of a process that will generate a Dataset/DataVersion
from a list of WorkflowExecutions.

Its initial form is a standalone script; once this is working, it will be refactored into API endpoints.

Author: Andrew Davison, CNRS, November 2024
"""

import sys
from enum import Enum
import io
import json
import math
import re
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

from ebrains_drive import BucketApiClient
from fairgraph.base import IRI
from fairgraph.client import KGClient
from fairgraph.kgproxy import KGProxy
from fairgraph.utility import as_list
import fairgraph.openminds.computation as omcmp
import fairgraph.openminds.core as omcore
import fairgraph.openminds.controlled_terms as terms
from jinja2 import Environment, select_autoescape, FileSystemLoader
from pydantic import AnyHttpUrl, BaseModel

# from ..auth.utils import get_kg_client_for_user_account
sys.path.append("/Users/adavison/dev/data/prov-api")
from provenance.auth.utils import get_kg_client_for_user_account


EBRAINS = KGProxy(
    omcore.Organization,
    "https://kg.ebrains.eu/api/instances/7dfdd91f-3d05-424a-80bd-6d1d5dc11cd3",
)
SWIFT_REPO = KGProxy(
    omcore.Organization,
    "https://kg.ebrains.eu/api/instances/877e7f6b-b9e6-4a07-a8c9-01f9f5e4ca8b",
)


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


license_pdfs = {
    License.ccby: "https://data-proxy.ebrains.eu/api/v1/buckets/d-c8395a2f-a6ae-40d0-ad0d-a87e8b9b610b/Licence-CC-BY.pdf"
}


class DOI:
    pass


def get_term(cls, term_name):
    # todo: retrieve the UUIDs from the KG
    if cls == terms.ProductAccessibility:
        term_uuid = {
            Accessibility.free: "b2ff7a47-b349-48d7-8ce4-cf51868675f1",
            Accessibility.underembargo: "897dc2af-405d-4df3-9152-6d9e5cae55d8",
            Accessibility.restricted: "459f153f-570c-46df-8b45-49ba56118d44",
        }[term_name]
    elif cls == terms.EthicsAssessment:
        term_uuid = {
            EthicsAssessment.not_required: "2386ab2b-9b4f-4e51-b8b9-c2acd110687a",
            EthicsAssessment.sensitive: "f660e0d6-285a-44f0-8837-edb3664ca462",
        }[term_name]
    elif cls == omcore.License:
        term_uuid = {
            License.cc0: "de5d8bfb-e1c3-405e-a6ce-ab0e5f74183c",
            License.ccby: "64c1704b-db12-4b29-9541-b9143d081044",
        }[term_name]
    else:
        raise NotImplementedError()
    return KGProxy(cls, f"https://kg.ebrains.eu/api/instances/{term_uuid}")


def get_workflow_execution(uuid: UUID, kg_client: KGClient) -> omcmp.WorkflowExecution:
    wfe = omcmp.WorkflowExecution.from_uuid(
        str(uuid),
        kg_client,
        follow_links={
            "recipe": {
                "developers": {"affiliations": {"member_of": {"has_parents": {}}}}
            },
            "started_by": {"affiliations": {"member_of": {"has_parents": {}}}},
        },
        scope="any",
    )
    if wfe:
        return wfe
    else:
        raise Exception("workflow execution not found")


def get_workflow_contributors(workflow_executions):
    # (1) recipe authors - TODO
    # (2) workflow execution started_by
    contributors = {}
    for wfe in workflow_executions:
        for contributor in as_list(wfe.started_by):
            contributors[contributor.uuid] = contributor
        for contributor in wfe.recipe.developers:
            contributors[contributor.uuid] = contributor
    return sorted(contributors.values(), key=lambda ctrb: ctrb.family_name)


def get_workflow_outputs(workflow_executions, kg_client):
    # do we return all files (both final outputs and intermediate files),
    # or just those defined in the workflow recipe as outputs?
    # maybe make this a choice for the user
    # for now, let's get only files from final stage
    outputs = []
    for wfe in workflow_executions:
        final_stage = as_list(wfe.stages)[-1].resolve(kg_client, scope="any")
        for output in as_list(final_stage.outputs):
            obj = output.resolve(kg_client, scope="any")
            if isinstance(obj, omcore.File):
                outputs.append(obj)
    return outputs


def get_workflow_recipes(workflow_executions, kg_client):
    recipes = {}
    for wfe in workflow_executions:
        recipe = wfe.recipe.resolve(
            kg_client, scope="any", follow_links={"repository": {}, "is_version_of": {}}
        )
        if not recipe.description:
            recipe.description = recipe.is_version_of.description
        recipes[recipe.uuid] = recipe
    return list(recipes.values())


def get_workflow_inputs(workflow_executions, kg_client):
    inputs = {}
    for wfe in workflow_executions:
        for stage in as_list(wfe.stages):
            stage = stage.resolve(kg_client, scope="any")
            for input in as_list(stage.inputs):
                obj = input.resolve(kg_client, scope="any")
                if isinstance(obj, omcore.File):
                    inputs[obj.uuid] = obj
    return list(inputs.values())


def get_workflow_config(workflow_executions, kg_client):
    common_inputs = {}
    variable_inputs = {}
    for wfe in workflow_executions:
        config = wfe.configuration.resolve(kg_client, scope="any")
        inputs = json.loads(config.configuration)
        for key, value in inputs.items():
            if key in variable_inputs:
                variable_inputs[key].append(value)
            elif key in common_inputs and common_inputs[key] != value:
                variable_inputs[key] = [common_inputs.pop(key), value]
            else:
                common_inputs[key] = value
    return common_inputs, variable_inputs


def get_timestamps(workflow_executions, kg_client):
    timestamps = []
    for wfe in workflow_executions:
        first_stage = wfe.stages[0].resolve(kg_client, scope="any")
        timestamps.append(first_stage.start_time)
    return timestamps


def get_techniques(workflow_executions, kg_client):
    techniques = {}
    for wfe in workflow_executions:
        for keyword in as_list(wfe.recipe.keywords):
            pass  # todo: filter by terms.AnalysisTechnique, terms.Technique
        for stage_recipe in as_list(wfe.recipe.has_parts):
            pass  # todo
    return list(techniques.values())


def find_parent_datasets(input_data, kg_client):
    datasets = {}
    for file_obj in input_data:
        repo = file_obj.file_repository.resolve(kg_client, scope="any")
        parents = repo.contains_content_of.resolve(kg_client, scope="any")
        for parent in as_list(parents):
            if isinstance(parent, omcore.DatasetVersion):
                datasets[parent.uuid] = parent
    return list(datasets.values())


def get_affiliations(author_nodes):
    organization_names = {}  # dict(str, int)
    author_names = {}  # dict(str, int)
    for au in author_nodes:
        author_names[au.full_name] = []
        for affil in as_list(au.affiliations):
            # todo: add logic for limiting affiliations based on timestamp
            #       cf live-papers
            org_name = affil.member_of.full_name
            if org_name in organization_names:
                org_index = organization_names[org_name]
            else:
                org_index = len(organization_names) + 1
                organization_names[org_name] = org_index
            author_names[au.full_name].append(org_index)
    return author_names, organization_names


def guess_copyright(author_nodes, year, kg_client):
    copyright_orgs = {}
    for au in author_nodes:
        for affil in as_list(au.affiliations):
            org = affil.member_of
            if org.has_parents:
                for parent_org in as_list(org.has_parents):
                    copyright_orgs[parent_org.uuid] = parent_org
            else:
                copyright_orgs[org.uuid] = org
    if copyright_orgs:
        return omcore.Copyright(holders=list(copyright_orgs.values()), years=year)
    else:
        return None


def format_file_size(qv):
    units = ["bytes", "kiB", "MiB", "GiB", "TiB", "PiB"]
    if qv:
        assert qv.unit.uuid == "6899d989-f510-43ad-b613-19c029b59ab2"  # bytes
        unit_index = math.floor(math.log2(qv.value) / 10)
        return f"{qv.value / math.pow(1024, unit_index):3.1f} {units[unit_index]}"
    else:
        return ""


def format_hash(hash_obj):
    if hash_obj:
        return f"{hash_obj.algorithm.lower()}${hash_obj.digest}"
    else:
        return ""


def generate_data_descriptor(
    uuid,
    name,
    description,
    authors,
    corresponding_author,
    files,
    version_spec,
    workflow_recipes,
    timestamps,
    input_data,
    common_inputs,
    variable_inputs,
    input_datasets,
    funding,
    license,
    kg_client,
):
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.realpath(__file__))),
        autoescape=select_autoescape(),
    )

    author_names, organization_names = get_affiliations(authors)

    file_info = [
        {"name": f"data_descriptor_{uuid.split('-')[-1]}.md"},
        {"name": license_pdfs[license].split("/")[-1]},
    ]
    for file in files:
        file_info.append(
            {
                "name": file.name,
                "size": format_file_size(file.storage_size),
                "hash": format_hash(file.hash),
            }
        )

    context = {
        "title": name,
        "summary": description,
        "authors": author_names,
        "affiliations": organization_names,
        "corresponding_author": corresponding_author.full_name,
        "corresponding_author_email": corresponding_author.contact_information.email,
        "files": file_info,
        "version_spec": version_spec,
        "workflow_recipes": workflow_recipes,
        "input_data": input_data,
        "timestamps": timestamps,
        "common_workflow_inputs": common_inputs,
        "variable_workflow_inputs": variable_inputs,
        "input_datasets": input_datasets,
        "acknowledgements": (
            (funding.acknowledgement or funding.award_title) if funding else ""
        ),
    }
    contents = env.get_template("data_descriptor_template.txt").render(context)
    return contents


def get_or_create_bucket(bucket_name, token):
    # do we need to create a collab also?
    client = BucketApiClient(token=token)
    try:
        bucket = client.buckets.get_bucket(bucket_name)
    except Exception:  # todo: specify ClientHttpError specifically
        # POST("https://data-proxy.ebrains.eu/api/v1/buckets/{bucket_name}/init")
        bucket = client.create_new(bucket_name)  # doesn't seem to work
    return bucket


def get_bucket_permalink(bucket, path):
    data = {
        "object_name": path,
        "bucket_name": bucket.name,
        "description": f"Permalink for {path} in bucket {bucket.name}, created by Provenance API",
    }
    response = bucket.client.post("/permalinks", json=data)
    if response.status_code not in (200, 201):
        raise Exception(response.text)
    else:
        entity_id = response.json()["entity_id"]
        return f"https://data-proxy.ebrains.eu/api/v1/permalinks/{entity_id}"


def upload_data_descriptor(data_descriptor_content, bucket):
    filename = f"data_descriptor_{bucket.name.split('-')[-1]}.md"
    fp = io.BytesIO(data_descriptor_content.encode("utf-8"))
    response = bucket.upload(fp, filename)
    # todo: get a public link for the data descriptor
    # return f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket.name}/{filename}"
    return get_bucket_permalink(bucket, filename)


def copy_outputs_to_bucket(bucket, workflow_outputs):
    for file_obj in workflow_outputs:
        assert isinstance(file_obj, omcore.File)
        url = str(file_obj.iri)
        assert url.startswith("https://data-proxy.ebrains.eu/api/v1")
        # todo: add .copy() method to DataproxyFile class in ebrains_storage package
        response = bucket.client.put(
            f"{url}/copy?to={bucket.name}&name={file_obj.name}", expected=(200, 201)
        )
        if response.status_code not in (200, 201):
            raise Exception(response.text)


def copy_license_to_bucket(bucket, license):
    source_url = license_pdfs[license]
    response = bucket.client.put(
        f"{source_url}/copy?to={bucket.name}", expected=(200, 201)
    )


def generate_slug(name):
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s


def collect(attr_name, objects):
    items = {}
    for obj in objects:
        for item in as_list(getattr(obj, attr_name)):
            items[item.uuid] = item
    return list(items.values())


def create_repository(bucket_name, workflow_outputs, kg_client):
    repo = omcore.FileRepository(
        name=f"bucket/{bucket_name}",
        content_type_patterns=None,  # todo: generate/retrieve from workflow_outputs
        format=None,  # (unless workflow generates a directory structure that corresponds to a content-type, e.g. BIDS)
        hash=None,  # generated by indexing processs?
        hosted_by=EBRAINS,
        iri=IRI(f"https://data-proxy.ebrains.eu/api/v1/buckets/{bucket_name}"),
        storage_size=None,  # generated by indexing process
        structure_pattern=None,
        type=SWIFT_REPO,
    )
    return repo


def create_dataset(
    dsv_uuid,
    workflow_executions: List[omcmp.WorkflowExecution],
    name,
    description,
    authors,
    custodian,
    repository,
    data_descriptor_url,
    accessibility,
    ethics_assessment,
    license,
    funding,
    homepage,
    how_to_cite,
    keywords,
    related_publications,
    input_data,
    input_datasets,
    version_spec,
    kg_client,
) -> Tuple[omcore.Dataset, omcore.DatasetVersion]:

    data_descriptor = omcore.WebResource(
        iri=IRI(data_descriptor_url),
        content_description=f"Data descriptor for dataset version #{dsv_uuid}",
        format=omcore.ContentType(
            name="text/markdown"
        ),  # todo: use KGProxy with hardcoded uuid for this content type
    )  # todo: save this

    year = "2024"  # todo: extract this from workflow_executions
    copyright = guess_copyright(authors, year, kg_client)

    dataset_version = omcore.DatasetVersion(
        id=f"https://kg.ebrains.eu/api/instances/{dsv_uuid}",
        accessibility=get_term(terms.ProductAccessibility, accessibility),
        authors=authors,
        behavioral_protocols=None,
        copyright=copyright,
        custodians=custodian,
        data_types=None,  # todo  SemanticDataType.derived_data
        description=description,  # todo: add some auto-generated material to the description?
        ethics_assessment=get_term(terms.EthicsAssessment, ethics_assessment),
        full_documentation=data_descriptor,
        full_name=name,
        funding=funding,
        homepage=None,  # todo
        how_to_cite=None,  # todo
        input_data=input_data,
        keywords=None,  # todo
        license=get_term(omcore.License, license),
        protocols=None,
        related_publications=None,  # todo
        repository=repository,
        short_name=generate_slug(name),
        studied_specimens=collect("studied_specimens", input_datasets),
        study_targets=collect("study_targets", input_datasets),
        support_channels=None,
        techniques=get_techniques(workflow_executions, kg_client),
        version_identifier="v1",
        version_innovation=version_spec,
        #experimental_approach?
    )

    dataset = omcore.Dataset(
        authors=dataset_version.authors,
        custodians=dataset_version.custodians,
        description=dataset_version.description,
        digital_identifier=None,
        full_name=dataset_version.full_name,
        has_versions=dataset_version,
        homepage=dataset_version.homepage,
        how_to_cite=dataset_version.how_to_cite,
        short_name=dataset_version.short_name,
    )

    return (dataset, dataset_version)


def save_to_kg(space, dataset, dataset_version, kg_client):
    # initialise KG space if necessary
    space_info = kg_client.space_info(space)
    if not space_info:
        kg_client.configure_space(
            space,
            [
                omcore.Person,
                omcore.Dataset,
                omcore.DatasetVersion,
                omcore.FileRepository,
                omcore.WebResource,
            ]
            + omcmp.list_kg_classes(),
        )
    # move workflow nodes to new space
    ## TODO
    # save data descriptor web resource
    dataset_version.full_documentation.save(kg_client, space=space, recursive=False)
    # save authors, if necessary
    ## TODO
    # save file repository
    dataset_version.repository.save(kg_client, space=space, recursive=False)
    # save dataset version
    dataset_version.save(kg_client, space=space, recursive=False)
    # save dataset
    dataset.save(kg_client, space=space, recursive=False)


def main(
    from_workflows: List[UUID],
    name: str,
    description: str,
    accessibility: Accessibility,
    ethics_assessment: EthicsAssessment,
    license: License,
    funding: Optional[str] = None,
    homepage: Optional[AnyHttpUrl] = None,
    how_to_cite: Optional[str] = None,
    keywords: Optional[
        List[str]
    ] = None,  # todo: define an Enum generated from openMINDS instances
    related_publications=Optional[List[DOI]],
    token: str = None,
):
    kg_client = get_kg_client_for_user_account(token)
    workflow_executions = [
        get_workflow_execution(uuid, kg_client) for uuid in from_workflows
    ]

    authors = get_workflow_contributors(workflow_executions)
    # we set the person creating the Dataset as the custodian / contact
    custodian = omcore.Person.me(kg_client, follow_links={"contact_information": {}})

    # dsv_uuid = str(uuid4())
    dsv_uuid = "9c654e77-7a1f-405f-91ea-832fe6168739"

    bucket_name = f"d-{dsv_uuid}"
    # bucket_name = "ebrains-2-0-wp4-task-4-3"
    bucket = get_or_create_bucket(bucket_name, token)

    workflow_outputs = get_workflow_outputs(workflow_executions, kg_client)
    copy_outputs_to_bucket(
        bucket, workflow_outputs
    )  # should this return modified File nodes?
    copy_license_to_bucket(bucket, license)

    repository = create_repository(bucket_name, workflow_outputs, kg_client)

    version_spec = "This is the first version of this dataset."
    workflow_recipes = get_workflow_recipes(workflow_executions, kg_client)
    timestamps = get_timestamps(workflow_executions, kg_client)
    input_data = get_workflow_inputs(workflow_executions, kg_client)
    input_datasets = find_parent_datasets(input_data, kg_client)
    for dsv in input_datasets:
        if not dsv.full_name:
            ds = dsv.is_version_of.resolve(kg_client, scope="any")
            dsv.full_name = ds.full_name
    common_inputs, variable_inputs = get_workflow_config(workflow_executions, kg_client)

    if funding:  # todo: could be list, for now we assume a single funding source
        funding = omcore.Funding.list(kg_client, award_title=funding, scope="any")
        if funding:
            funding = funding[0]

    data_descriptor_content = generate_data_descriptor(
        dsv_uuid,
        name,
        description,
        authors,
        custodian,
        workflow_outputs,
        version_spec,
        workflow_recipes,
        timestamps,
        input_data,
        common_inputs,
        variable_inputs,
        input_datasets,
        funding,
        license,
        kg_client,
    )
    print(data_descriptor_content)
    data_descriptor_url = upload_data_descriptor(data_descriptor_content, bucket)
    # data_descriptor_url = "http://example.com/data_descriptor.md"

    dataset, dataset_version = create_dataset(
        dsv_uuid,
        workflow_executions,
        name,
        description,
        authors,
        custodian,
        repository,
        data_descriptor_url,
        accessibility,
        ethics_assessment,
        license,
        funding,
        homepage,
        how_to_cite,
        keywords,
        related_publications,
        input_data,
        input_datasets,
        version_spec=version_spec,
        kg_client=kg_client,
    )

    # save dataset and dataset_version to KG
    save_to_kg(f"collab-{bucket_name}", dataset, dataset_version, kg_client)


if __name__ == "__main__":
    import os

    token = os.environ["EBRAINS_AUTH_TOKEN"]
    main(
        # from_workflows=[
        #    "09251605-c7f4-463b-bbc1-bbf0baeeb382",
        #    "428a3135-bfb6-4c66-8b69-8fa383c736be",
        # ],  # sc3 example
        # from_workflows=["93427a9c-36fd-4bdf-b4e3-1c4e55045786"],  # PSD example
        # from_workflows=[
        #    "a8f78f51-e875-400c-bc8e-30179fefda5d",
        #    "a0a8ea01-4934-4c73-bb51-dbfe8b65beaf",
        # ],  # demonstrator_workflow-001 example
        from_workflows=[
            "d3ab0bde-1737-4271-89f0-d90fc9f02561",
            "7257c759-3bf0-455e-8ad3-9af8c028078d",
        ],  # demonstrator_workflow-001 example
        # name="Test dataset for M4.2",
        name="Wavelet analysis of multi-electrode recordings of macaque motor cortex during an instructed delayed reach-to-grasp task",
        description=(
            "A dataset generated entirely from running EBRAINS data processing workflows, "
            "with metadata auto-generated from workflow provenance recording.\n\n"
            "This is a proof-of-concept, corresponding to Milestone 4.2 in the EBRAINS-2.0 project.\n"
            "The example workflow involved downloading experimental data from an EBRAINS dataset, "
            "running a sequence of data analysis steps with Elephant (Butterworth filter then wavelet transform), "
            "then uploading the results to EBRAINS Bucket storage."
        ),
        accessibility=Accessibility.free,
        ethics_assessment=EthicsAssessment.not_required,
        funding="EBRAINS 2.0",
        license=License.ccby,
        token=token,
    )
