
import sys
import os
from uuid import uuid4
import json
from datetime import datetime, timezone
import pytest
from pydantic import parse_obj_as
from fastapi.testclient import TestClient

from fairgraph.client import KGClient
import fairgraph.openminds.core as omcore
import fairgraph.openminds.controlledterms as omterms
import fairgraph.openminds.computation as omcmp
from fairgraph.base import IRI, as_list

sys.path.append(".")  # run tests in root directory of project
from provenance.main import app

ID_PREFIX = "https://kg.ebrains.eu/api/instances"
TEST_SPACE = "collab-provenance-api-development"
#TEST_SPACE = "collab-ebrains-workflows"



kg_client = KGClient(host="core.kg-ppd.ebrains.eu")  # don't use production for testing
assert os.environ["KG_CORE_API_HOST"] == kg_client.host
if kg_client.user_info():
    have_kg_connection = True
else:
    have_kg_connection = False


test_client = TestClient(app)


no_kg_err_msg = "No KG connection - have you set the environment variable KG_AUTH_TOKEN?"


@pytest.fixture(scope="module")
def units():
    return {
        item.name: item
        for item in omterms.UnitOfMeasurement.list(kg_client, api="core", size=1000, scope="in progress")
    }


@pytest.fixture(scope="module")
def person_obj():
    obj = omcore.Person(
        id=f"{ID_PREFIX}/{uuid4()}",
        family_name="Baggins",
        given_name="Bilbo",
        digital_identifiers=[
            omcore.ORCID(
                id=f"{ID_PREFIX}/{uuid4()}",
                identifier="http://orcid.org/0000-0002-4793-7541"
            )
        ]
    )
    obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield obj
    obj.digital_identifiers[0].delete(kg_client)
    obj.delete(kg_client)


@pytest.fixture(scope="module")
def input_file_obj(units):
    obj = omcore.File(
        id=f"{ID_PREFIX}/{uuid4()}",
        content_description="Demonstration data for validation framework",
        format=omcore.ContentType(name="application/json"),
        hash=omcore.Hash(algorithm="SHA-1", digest="716c29320b1e329196ce15d904f7d4e3c7c46685"),
        iri=IRI("https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json"),
        name="InputResistance_data.json",
        storage_size=omcore.QuantitativeValue(value=34.0, unit=units["byte"])
    )
    obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield obj
    obj.delete(kg_client)


@pytest.fixture(scope="module")
def output_file_objs(units):
    objs = [
        omcore.File(
            id=f"{ID_PREFIX}/{uuid4()}",
            content_description="File generated by some computation",
            format=omcore.ContentType(name="image/png"),
            hash=omcore.Hash(algorithm="SHA-1", digest="9006f7ca30ee32d210249ba125dfd96d18b6669e"),
            iri=IRI("https://drive.ebrains.eu/f/61ceb5c4aa3c4468a26c/"),
            name="output_files/Freund_SGA1_T1.2.5_HC-awake-ephys_HBP_1_cell1_ephys__160712_cell1_LFP.png",
            storage_size=omcore.QuantitativeValue(value=60715.0, unit=units["byte"])
        ),
        omcore.File(
            id=f"{ID_PREFIX}/{uuid4()}",
            content_description="File generated by a simulation",
            format=omcore.ContentType(name="application.vnd.nwb.nwbn+hdf"),  # there is a typo in this name, fixed on Github, waiting for it to be fixed in KG
            hash=omcore.Hash(algorithm="SHA-1", digest="a006f7ca30ee32d210249ba125dfd96d18b6669f"),
            iri=IRI("https://gpfs-proxy.brainsimulation.eu/cscs/myproject/output_data/simulation_results.nwb"),
            name="output_data/simulation_results.nwb",
            storage_size=omcore.QuantitativeValue(value=605888.0, unit=units["byte"])
        ),
    ]
    for obj in objs:
        obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield objs
    for obj in objs:
        obj.delete(kg_client)


@pytest.fixture(scope="module")
def software_version_objs():
    objs = [
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="Elephant", alias="Elephant", version_identifier="0.10.0"),
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="numpy", alias="numpy", version_identifier="1.19.3"),
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="neo", alias="neo", version_identifier="0.9.0"),
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="spyNNaker", alias="spyNNaker", version_identifier="5.0.0"),
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="NEST", alias="nest", version_identifier="3.1.0"),
        omcore.SoftwareVersion(id=f"{ID_PREFIX}/{uuid4()}", name="MyScript", alias="myscript", version_identifier="0.0.1"),
    ]
    for obj in objs:
        obj.save(kg_client, space=TEST_SPACE)
    yield objs
    for obj in objs:
        obj.delete(kg_client)


@pytest.fixture(scope="module")
def model_version_obj():
    obj = omcore.ModelVersion(
        id=f"{ID_PREFIX}/{uuid4()}",
        name="A really good model",
        alias="a-really-good-model",
        version_identifier="10.0"
    )
    obj.save(kg_client, space=TEST_SPACE)
    yield obj
    obj.delete(kg_client)


@pytest.fixture(scope="module")
def hardware_obj():
    obj = omcmp.HardwareSystem.by_name("CSCS Castor", kg_client, scope="in progress", space="common")
    assert obj is not None
    return obj


@pytest.fixture(scope="module")
def environment_obj(software_version_objs, hardware_obj):
    obj = omcmp.Environment(
        id=f"{ID_PREFIX}/{uuid4()}",
        name="Some environment that doesn't really exist",
        hardware=hardware_obj,
        configuration=omcore.Configuration(
            configuration=json.dumps({
                "parameter1": "value1",
                "parameter2": "value2"
            }, indent=2),
            lookup_label="hardware configuration for fake hardware",
            format=omcore.ContentType(name="application/json")
        ),
        software=software_version_objs[1:4],
        description="Default environment on fake hardware"

    )
    obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield obj
    obj.delete(kg_client)


@pytest.fixture(scope="module")
def launch_config_obj():
    obj = omcmp.LaunchConfiguration(
        id=f"{ID_PREFIX}/{uuid4()}",
        executable="/usr/bin/python",
        name="dummy launch config",
        arguments=["-Werror"],
        environment_variables=omcore.PropertyValueList(
            lookup_label="Dummy environment variables for testing",
            property_value_pairs=[omcore.StringProperty(name="COLLAB_ID", value=TEST_SPACE)]
        )
    )
    obj.save(kg_client, space=TEST_SPACE)
    yield obj
    obj.delete(kg_client)


@pytest.fixture(scope="class")
def data_analysis_obj(person_obj, input_file_obj, output_file_objs, software_version_objs,
                      environment_obj, launch_config_obj, units):
    timestamp = datetime.now()
    resource_usage = [omcore.QuantitativeValue(value=2.0,
                                               unit=units["hour"])]
    obj = omcmp.DataAnalysis(
        id=f"{ID_PREFIX}/{uuid4()}",
        lookup_label=f"Test-{timestamp.isoformat()}",
        inputs=[input_file_obj, software_version_objs[0]],
        outputs=output_file_objs[0:1],
        environment=environment_obj,
        launch_configuration=launch_config_obj,
        start_time=datetime(2021, 5, 28, 16, 32, 58, 597000, tzinfo=timezone.utc),
        end_time=datetime(2021, 5, 28, 18, 32, 58,  597000, tzinfo=timezone.utc),
        started_by=person_obj,
        status=omterms.ActionStatusType(name="potential"),  # i.e. queued
        resource_usages=resource_usage,
        tags=["string"]
    )
    obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield obj
    # teardown
    obj.delete(kg_client)


@pytest.fixture(scope="class")
def visualisation_obj(person_obj, input_file_obj, output_file_objs, software_version_objs,
                      environment_obj, launch_config_obj, units):
    resource_usage = [omcore.QuantitativeValue(value=29.0,
                                               unit=units["second"])]
    obj = omcmp.Visualization(
        id=f"{ID_PREFIX}/{uuid4()}",
        inputs=[input_file_obj, software_version_objs[0]],
        outputs=output_file_objs[0:1],
        environment=environment_obj,
        launch_configuration=launch_config_obj,
        start_time=datetime(2021, 6, 28, 16, 32, 58, tzinfo=timezone.utc),
        end_time=datetime(2021, 6, 28, 16, 33, 27, tzinfo=timezone.utc),
        started_by=person_obj,
        status=omterms.ActionStatusType(name="potential"),
        resource_usages=resource_usage,
        tags=["string"]
    )
    obj.save(kg_client, space=TEST_SPACE, recursive=True)
    yield obj
    # teardown
    obj.delete(kg_client)


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
def test_about():
    response = test_client.get("/")
    assert response.status_code == 200
    expected = {
        'about': 'This is the EBRAINS Provenance API.',
        'links': {'documentation': '/docs'}
    }
    assert response.json() == expected


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestFixtures:
    def test_fixtures(self, data_analysis_obj):
        assert data_analysis_obj.started_by.given_name == "Bilbo"
        assert data_analysis_obj.end_time == datetime(2021, 5, 28, 18, 32, 58,  597000, tzinfo=timezone.utc)


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestGetDataAnalysis:

    def test_get_data_analysis(self, data_analysis_obj, environment_obj, software_version_objs):
        token = kg_client.token
        response = test_client.get(f"/analyses/{data_analysis_obj.uuid}",
                                headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

        expected = {
            "description": None,
            "end_time": "2021-05-28T18:32:58.597000+00:00",
            "environment": {"configuration": {
                                "parameter1": "value1",
                                "parameter2": "value2"
                            },
                            "description": "Default environment on fake hardware",
                            "hardware": "CSCS Castor",
                            "id": environment_obj.uuid,
                            "name": "Some environment that doesn't really exist",
                            "software": [
                                {"id": obj.uuid, "software_name": obj.name, "software_version": obj.version_identifier}
                                for obj in software_version_objs[1:4]
                            ]},
            "id": data_analysis_obj.uuid,
            "input": [{"description": "Demonstration data for validation framework",
                    "file_name": "InputResistance_data.json",
                    "format": "application/json",
                    "hash": {"algorithm": "SHA-1",
                                "value": "716c29320b1e329196ce15d904f7d4e3c7c46685"},
                    "location": "https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json",
                    "size": 34},
                    {"id": software_version_objs[0].uuid,
                    "software_name": "Elephant",
                    "software_version": "0.10.0"}],
            "launch_config": {"arguments": ["-Werror"],
                            "description": None,
                            "environment_variables": {"description": "Dummy environment variables for testing",
                                                        "items": [{"name": "COLLAB_ID",
                                                                "value": TEST_SPACE}]},
                            "executable": "/usr/bin/python",
                            "name": "dummy launch config"},
            "output": [{"description": "File generated by some computation",
                        "file_name": "output_files/Freund_SGA1_T1.2.5_HC-awake-ephys_HBP_1_cell1_ephys__160712_cell1_LFP.png",
                        "format": "image/png",
                        "hash": {"algorithm": "SHA-1",
                                "value": "9006f7ca30ee32d210249ba125dfd96d18b6669e"},
                        "location": "https://drive.ebrains.eu/f/61ceb5c4aa3c4468a26c/",
                        "size": 60715}],
            "recipe_id": None,
            "resource_usage": [{"units": "hour", "value": 2.0}],
            "start_time": "2021-05-28T16:32:58.597000+00:00",
            "started_by": {"family_name": "Baggins",
                        "given_name": "Bilbo",
                        "orcid": "http://orcid.org/0000-0002-4793-7541"},
            "status": "queued",
            "tags": ["string"],
            "type": "data analysis"
        }

        assert response.json() == expected


    def test_query_data_analysis_by_inputs(self, data_analysis_obj, input_file_obj):
        token = kg_client.token
        response_query = test_client.get(f"/analyses/?input_data={input_file_obj.uuid}&space={TEST_SPACE}",
                                        headers={"Authorization": f"Bearer {token}"})
        assert response_query.status_code == 200
        response_direct = test_client.get(f"/analyses/{data_analysis_obj.uuid}",
                                        headers={"Authorization": f"Bearer {token}"})
        assert response_query.json() == [response_direct.json()]


    def test_query_data_analysis_by_simulation(self, data_analysis_obj, input_file_obj):
        pass


    def test_query_data_analysis_by_software(self, data_analysis_obj, software_version_objs):
        token = kg_client.token
        # software that was explicitly an input
        response_query1 = test_client.get(f"/analyses/?software={software_version_objs[0].uuid}&space={TEST_SPACE}",
                                          headers={"Authorization": f"Bearer {token}"})
        assert response_query1.status_code == 200

        response_direct = test_client.get(f"/analyses/{data_analysis_obj.uuid}",
                                          headers={"Authorization": f"Bearer {token}"})
        assert response_query1.json() == [response_direct.json()]

        # software that was in the environment
        response_query2 = test_client.get(f"/analyses/?software={software_version_objs[3].uuid}&space={TEST_SPACE}",
                                          headers={"Authorization": f"Bearer {token}"})
        assert response_query2.status_code == 200
        assert response_query1.json() == [response_direct.json()]

        # software that wasn't used
        response_query3 = test_client.get(f"/analyses/?software={software_version_objs[4].uuid}&space={TEST_SPACE}",
                                          headers={"Authorization": f"Bearer {token}"})
        assert response_query3.status_code == 200
        assert response_query3.json() == []

    def test_query_data_analysis_by_platform(self, data_analysis_obj):
        token = kg_client.token
        response = test_client.get(f"/analyses/?platform=CSCS%20Castor&space={TEST_SPACE}",
                                headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert all(item["environment"]["hardware"] == "CSCS Castor" for item in data)

    def test_query_data_analysis_by_status(self, data_analysis_obj, input_file_obj):
        token = kg_client.token
        response = test_client.get(f"/analyses/?status=queued&space={TEST_SPACE}",
                                headers={"Authorization": f"Bearer {token}"})
        data = response.json()
        assert len(data) > 0
        assert all(item["status"] == "queued" for item in data)

        response = test_client.get(f"/analyses/?status=completed&space={TEST_SPACE}",
                                headers={"Authorization": f"Bearer {token}"})
        data = response.json()
        assert len(data) == 0


    def test_query_data_analysis_by_tags(self, data_analysis_obj, input_file_obj):
        pass


    def test_query_data_analysis_by_dataset(self, data_analysis_obj, input_file_obj):
        pass


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestModifyDataAnaysis:

    def test_patch_data_analysis(self, data_analysis_obj):
        token = kg_client.token
        response = test_client.patch(f"/analyses/{data_analysis_obj.uuid}",
                                    json={"status": "completed"},
                                    headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

        response2 = test_client.get(f"/analyses/{data_analysis_obj.uuid}",
                                    headers={"Authorization": f"Bearer {token}"})
        assert response2.status_code == 200
        assert response2.json()["status"] == "completed"



@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestGetVisualisation:

    def test_get_visualisation(self, visualisation_obj, environment_obj, software_version_objs):
        token = kg_client.token
        response = test_client.get(f"/visualisations/{visualisation_obj.uuid}",
                                headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        expected = {
            "description": None,
            "end_time": "2021-06-28T16:33:27+00:00",
            "environment": {"configuration": {
                                "parameter1": "value1",
                                "parameter2": "value2"
                            },
                            "description": "Default environment on fake hardware",
                            "hardware": "CSCS Castor",
                            "id": environment_obj.uuid,
                            "name": "Some environment that doesn't really exist",
                            "software": [
                                {"id": obj.uuid, "software_name": obj.name, "software_version": obj.version_identifier}
                                for obj in software_version_objs[1:4]
                            ]},
            "id": visualisation_obj.uuid,
            "input": [{"description": "Demonstration data for validation framework",
                    "file_name": "InputResistance_data.json",
                    "format": "application/json",
                    "hash": {"algorithm": "SHA-1",
                                "value": "716c29320b1e329196ce15d904f7d4e3c7c46685"},
                    "location": "https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json",
                    "size": 34},
                    {"id": software_version_objs[0].uuid,
                    "software_name": "Elephant",
                    "software_version": "0.10.0"}],
            "launch_config": {"arguments": ["-Werror"],
                            "description": None,
                            "environment_variables": {"description": "Dummy environment variables for testing",
                                                        "items": [{"name": "COLLAB_ID",
                                                                "value": TEST_SPACE}]},
                            "executable": "/usr/bin/python",
                            "name": "dummy launch config"},
            "output": [{"description": "File generated by some computation",
                        "file_name": "output_files/Freund_SGA1_T1.2.5_HC-awake-ephys_HBP_1_cell1_ephys__160712_cell1_LFP.png",
                        "format": "image/png",
                        "hash": {"algorithm": "SHA-1",
                                "value": "9006f7ca30ee32d210249ba125dfd96d18b6669e"},
                        "location": "https://drive.ebrains.eu/f/61ceb5c4aa3c4468a26c/",
                        "size": 60715}],
            "recipe_id": None,
            "resource_usage": [{"units": "second", "value": 29.0}],
            "start_time": "2021-06-28T16:32:58+00:00",
            "started_by": {"family_name": "Baggins",
                        "given_name": "Bilbo",
                        "orcid": "http://orcid.org/0000-0002-4793-7541"},
            "status": "queued",
            "tags": ["string"],
            "type": "visualization"
        }

        assert response.json() == expected


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestModifyVisualisation:

    def test_patch_visualisation(self, visualisation_obj):
        token = kg_client.token
        response = test_client.patch(f"/visualisations/{visualisation_obj.uuid}",
                                    json={"status": "failed"},
                                    headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

        response2 = test_client.get(f"/visualisations/{visualisation_obj.uuid}",
                                    headers={"Authorization": f"Bearer {token}"})
        assert response2.status_code == 200
        assert response2.json()["status"] == "failed"


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestCreateSimulation:

    def test_post_simulation(self, person_obj, output_file_objs, software_version_objs,
                             model_version_obj, environment_obj, launch_config_obj):
        data = {
            "environment": {
                "configuration": json.loads(environment_obj.configuration.configuration),
                "description": environment_obj.description,
                "hardware": environment_obj.hardware.name,
                "id": environment_obj.uuid,
                "name": environment_obj.name,
                "software": [
                    {"id": obj.uuid, "software_name": obj.name, "software_version": obj.version_identifier}
                    for obj in software_version_objs[1::3]
                ]
            },
            "input": [
                {
                    "id": software_version_objs[5].uuid,
                    "software_name": software_version_objs[5].name,
                    "software_version": software_version_objs[5].version_identifier
                },
                {
                    "model_version_id": model_version_obj.uuid
                }
            ],
            "launch_config": {
                "arguments": launch_config_obj.arguments,
                "environment_variables": {
                    "items": [{"name": envvar.name, "value": envvar.value}
                            for envvar in launch_config_obj.environment_variables.property_value_pairs]
                },
                "executable": launch_config_obj.executable,
                "name": launch_config_obj.name
            },
            "output": [
                {
                    "description": output_file_objs[1].content_description,
                    "file_name": output_file_objs[1].name,
                    "format": output_file_objs[1].format.name,
                    "hash": {
                        "algorithm": output_file_objs[1].hash.algorithm,
                        "value": output_file_objs[1].hash.digest,
                    },
                    "location": str(output_file_objs[1].iri),
                    "size": output_file_objs[1].storage_size.value
                }
            ],
            "start_time": datetime.now().isoformat(),
            "started_by": {
                "family_name": person_obj.family_name,
                "given_name": person_obj.given_name,
                "orcid": person_obj.digital_identifiers[0].identifier},
            "status": "queued",
            "tags": ["ham", "eggs"],
            "type": "simulation"
        }
        token = kg_client.token
        response = test_client.post(f"/simulations/?space={TEST_SPACE}",
                                    json=data,
                                    headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 201



@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestCreateWorkflowRecipe:

    def test_post_recipe(self, person_obj):
        existing_recipe = omcmp.WorkflowRecipeVersion.list(kg_client, alias="PSD_workflow_KG", version_identifier="12345678", scope="in progress")
        if existing_recipe:
            for recipe in as_list(existing_recipe):
                recipe.delete(kg_client)
        data = {
            "name": "PSD (Power Spectral Density) Calculation Workflow with input file from Knowledge Graph",
            "alias": "PSD_workflow_KG",
            "custodians": [{"given_name": person_obj.given_name, "family_name": person_obj.family_name}],
            "description": "description goes here",
            "developers": [{"given_name": person_obj.given_name, "family_name": person_obj.family_name}],
            "homepage": "https://gitlab.ebrains.eu/technical-coordination/project-internal/workflows/cwl-workflows/-/tree/main/PSD_workflow_KG",
            "location": "https://gitlab.ebrains.eu/technical-coordination/project-internal/workflows/cwl-workflows",
            "version_identifier": "12345678"
        }
        token = kg_client.token
        response = test_client.post("/recipes/?space=collab-provenance-api-development",
                                    json=data,
                                    headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 201
        # todo: delete the new recipe again
