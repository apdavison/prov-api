
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
import fairgraph.openminds.controlled_terms as omterms
import fairgraph.openminds.computation as omcmp
from fairgraph import IRI
from fairgraph.utility import as_list

sys.path.append(".")  # run tests in root directory of project
from provenance.main import app

ID_PREFIX = "https://kg.ebrains.eu/api/instances"



kg_client = KGClient(host="core.kg.ebrains.eu", allow_interactive=False)  # DANGER: using prod client - only put GET tests in this file
if kg_client.user_info() and os.environ["KG_CORE_API_HOST"] == kg_client.host:
    have_kg_connection = True
else:
    have_kg_connection = False


test_client = TestClient(app)


no_kg_err_msg = "No KG connection - have you set the environment variables KG_AUTH_TOKEN and KG_CORE_API_HOST?"


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQueryWorkflowRecipes:

    def test_query_all_in_computation_space(self):
        response = test_client.get("/recipes/?space=computation",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200

    def test_query_all(self):
        response = test_client.get("/recipes/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQuerySimulations:

    def test_query_all(self):
        response = test_client.get("/simulations/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQueryDataAnalyses:

    def test_query_all(self):
        response = test_client.get("/analyses/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQueryVisualisations:

    def test_query_all(self):
        response = test_client.get("/visualisations/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQueryDataCopies:

    def test_query_all(self):
        response = test_client.get("/datacopies/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200


@pytest.mark.skipif(not have_kg_connection, reason=no_kg_err_msg)
class TestQueryWorkflowExecutions:

    def test_query_all(self):
        response = test_client.get("/workflows/",
                                    headers={"Authorization": f"Bearer {kg_client.token}"})
        assert response.status_code == 200
