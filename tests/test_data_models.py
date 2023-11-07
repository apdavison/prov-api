
import sys
from copy import deepcopy
from datetime import datetime, timezone
from uuid import UUID
import json
from fairgraph.utility import compact_uri
from fairgraph.openminds.core.miscellaneous.quantitative_value import QuantitativeValue

import jsondiff

from pydantic import parse_obj_as

sys.path.append(".")
from provenance.common.data_models import ResourceUsage, get_repository_iri
from provenance.dataanalysis.data_models import DataAnalysis
from provenance.visualisation.data_models import Visualisation
from provenance.optimisation.data_models import Optimisation
from provenance.simulation.data_models import Simulation
from provenance.workflows.data_models import WorkflowExecution
import provenance.common.examples
import provenance.simulation.examples
import provenance.dataanalysis.examples
import provenance.visualisation.examples
import provenance.optimisation.examples
import provenance.workflows.examples
import fairgraph.openminds.core as omcore
import fairgraph.openminds.controlledterms as omterms
import fairgraph.openminds.computation as omcmp
from fairgraph import IRI


EXAMPLES = provenance.common.examples.EXAMPLES
EXAMPLES.update(provenance.dataanalysis.examples.EXAMPLES)
EXAMPLES.update(provenance.visualisation.examples.EXAMPLES)
EXAMPLES.update(provenance.optimisation.examples.EXAMPLES)
EXAMPLES.update(provenance.simulation.examples.EXAMPLES)
EXAMPLES.update(provenance.workflows.examples.EXAMPLES)

ID_PREFIX = "https://kg.ebrains.eu/api/instances"


class MockKGClient:

    def user_info(self):
        return {
            "http://schema.org/familyName": "Holmes",
            "http://schema.org/givenName": "Sherlock"
        }

    def list(self, cls, space=None, from_index=0, size=100, api="core", scope="released",
             resolved=False, filter=None):
        if cls == omcore.Person:
            return [omcore.Person(given_name="Sherlock", family_name="Holmes")]
        return None

    def uri_from_uuid(self, uuid):
        return f"{ID_PREFIX}/{uuid}"

    def uuid_from_uri(self, uri):
        return uri.split("/")[-1]

    def instance_from_full_uri(self, uri, use_cache=True, scope="released", resolved=False):
        objects = {
            "kg:3fa85f64-5717-4562-b3fc-2c963f66afa6": {
                "@id": "kg:3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "@type": ["https://openminds.ebrains.eu/core/ModelVersion"],
                "@context": {"vocab": "https://openminds.ebrains.eu/vocab/", "kg": "https://kg.ebrains.eu/api/instances/"},
                'vocab:format': {'@id': "kg:47fd24aa-69c5-4e42-a6ca-a6ade6595a14",
                                 '@type': ['https://openminds.ebrains.eu/core/ContentType']},
                'vocab:license': {'@id': "kg:c47bf382-d44b-4a2d-98a3-40ca18e6911d",
                                  '@type': ['https://openminds.ebrains.eu/core/License']},
                'vocab:accessibility': {'@id': "kg:b2ff7a47-b349-48d7-8ce4-cf51868675f1",
                                        '@type': ['https://openminds.ebrains.eu/controlledTerms/ProductAccessibility']},
                'vocab:fullDocumentation': {'@id': 'http://example.com',
                                            '@type': ['https://openminds.ebrains.eu/core/WebResource']},
                'vocab:fullName': 'fake model for testing',
                'vocab:releaseDate': '1999-12-31T23:59:59',
                'vocab:shortName': 'fake-model-for-testing',
                'vocab:versionIdentifier': '1.0',
                'vocab:versionInnovation': 'new stuff'
            }
        }
        compacted_uri = compact_uri(uri, {"kg": "https://kg.ebrains.eu/api/instances/"})
        return objects.get(compacted_uri, None)


class TestCommon:

    def test_get_repository_iri(self):
        file_iri = "https://object.cscs.ch/v1/AUTH_63ea6845b1d34ad7a43c8158d9572867/Freund_SGA1_T1.2.5/HC-awake-ephys/HBP_1/cell1/ephys/Freund_SGA1_T1.2.5_HC-awake-ephys_HBP_1_cell1_ephys__160712_cell1.smr"
        repo_iri = str(get_repository_iri(file_iri))
        assert repo_iri == "https://object.cscs.ch/v1/AUTH_63ea6845b1d34ad7a43c8158d9572867/Freund_SGA1_T1.2.5"

        file_iri = "https://gpfs-proxy.brainsimulation.eu/cscs/myproject/output_data/simulation_results.nwb"
        repo_iri = str(get_repository_iri(file_iri))
        assert repo_iri == "https://gpfs-proxy.brainsimulation.eu/cscs/myproject"

    def test_resource_usage(self):
        pydantic_obj = parse_obj_as(ResourceUsage, EXAMPLES["ResourceUsage"])
        kg_client = MockKGClient()
        kg_object = pydantic_obj.to_kg_object(kg_client)
        assert isinstance(kg_object, QuantitativeValue)

    def test_numerical_parameter(self):
        pass


class TestDataAnalysis:

    def test_conversion_to_kg_objects(self):
        #pydantic_obj = DataAnalysis(**EXAMPLES["DataAnalysis"])
        pydantic_obj = parse_obj_as(DataAnalysis, EXAMPLES["DataAnalysis"])
        kg_client = MockKGClient()
        kg_objects = pydantic_obj.to_kg_object(kg_client)

    def test_conversion_from_kg_objects(self):

        started_by = omcore.Person(
            family_name="Destexhe",
            given_name="Alain",
            digital_identifiers=[omcore.ORCID(identifier="https://orcid.org/0000-0001-7405-0455")],
            id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"
        )
        inputs = [
            omcore.File(
                content_description="Demonstration data for validation framework",
                format=omcore.ContentType(name="application/json", id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"),
                hash=omcore.Hash(algorithm="SHA-1", digest="716c29320b1e329196ce15d904f7d4e3c7c46685"),
                iri=IRI("https://object.cscs.ch/v1/AUTH_c0a333ecf7c045809321ce9d9ecdfdea/VF_paper_demo/obs_data/InputResistance_data.json"),
                name="InputResistance_data.json",
                storage_size=omcore.QuantitativeValue(value=34.0, unit=omterms.UnitOfMeasurement(name="byte"))
            ),
            omcore.SoftwareVersion(
                name="Elephant",
                version_identifier="0.10.0",
                id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"
            )
        ]
        outputs = [deepcopy(inputs[0])]
        environment = omcmp.Environment(
            id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000",
            name="SpiNNaker default 2021-10-13",
            hardware=omcmp.HardwareSystem(name="SpiNNaker"),
            configuration=omcore.Configuration(
                    configuration=json.dumps(
                        {
                            "parameter1": "value1",
                            "parameter2": "value2"
                        },
                        indent=2
                    ),
                    format=omcore.ContentType(name="application/json", id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"),
                    lookup_label="hardware configuration for SpiNNaker 1M core machine"
            ),
            software=[
                omcore.SoftwareVersion(name="numpy", version_identifier="1.19.3", id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"),
                omcore.SoftwareVersion(name="neo", version_identifier="0.9.0", id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000"),
                omcore.SoftwareVersion(name="spyNNaker", version_identifier="5.0.0", id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000")
            ],
            description="Default environment on SpiNNaker 1M core machine as of 2020-10-13 (not really, this is just for example purposes)."

        )
        launch_configuration = omcmp.LaunchConfiguration(
            executable="/usr/bin/python",
            arguments=["-Werror"],
            environment_variables=omcore.PropertyValueList(
                property_value_pairs=[omcore.StringProperty(name= "COLLAB_ID", value= "myspace")]
            )
        )
        resource_usage = [omcore.QuantitativeValue(value=1017.3, unit=omterms.UnitOfMeasurement(name="core-hour"))]
        data_analysis = omcmp.DataAnalysis(
            id=f"{ID_PREFIX}/00000000-0000-0000-0000-000000000000",
            inputs=inputs,
            outputs=outputs,
            environment=environment,
            launch_configuration=launch_configuration,
            start_time=datetime(2021, 5, 28, 16, 32, 58, 597000, tzinfo=timezone.utc),
            end_time=datetime(2021, 5, 28, 16, 32, 58,  597000, tzinfo=timezone.utc),
            started_by=started_by,
            status=omterms.ActionStatusType(name="potential"),  # i.e. queued
            resource_usages=resource_usage,
            tags=["string"]
        )

        client = MockKGClient()
        pydantic_obj = DataAnalysis.from_kg_object(data_analysis, client)

        # remove IDs added by from_kg_object() for the comparison
        pydantic_obj.environment.id = None
        for item in pydantic_obj.environment.software:
            item.id = None
        pydantic_obj.input[1].id = None

        actual = json.loads(pydantic_obj.json(exclude_none=True))
        #expected = json.dumps(deepcopy(EXAMPLES["DataAnalysis"]))
        expected = EXAMPLES["DataAnalysis"]

        #diff = jsondiff.diff(actual, expected, load=True, syntax="explicit")
        #assert len(diff) == 0
        assert actual == expected


class TestVisualisation:

    def test_conversion_to_kg_objects(self):
        pydantic_obj = parse_obj_as(Visualisation, EXAMPLES["Visualisation"])
        kg_client = MockKGClient()
        kg_objects = pydantic_obj.to_kg_object(kg_client)


class TestSimulation:

    def test_conversion_to_kg_objects(self):
        pydantic_obj = parse_obj_as(Simulation, EXAMPLES["Simulation"])
        kg_client = MockKGClient()
        kg_objects = pydantic_obj.to_kg_object(kg_client)


class TestOptimisation:

    def test_conversion_to_kg_objects(self):
        pydantic_obj = parse_obj_as(Optimisation, EXAMPLES["Optimisation"])
        #kg_client = MockKGClient()
        #kg_objects = pydantic_obj.to_kg_object(kg_client)


class TestWorkflow:

    def test_conversion_to_kg_objects(self):
        pydantic_obj = parse_obj_as(WorkflowExecution, EXAMPLES["WorkflowExecution"])
        kg_client = MockKGClient()
        kg_objects = pydantic_obj.to_kg_object(kg_client)
