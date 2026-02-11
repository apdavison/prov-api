import sys
from fastapi.testclient import TestClient

sys.path.append(".")  # run tests in root directory of project
from provenance.main import app

test_client = TestClient(app)



def test_openapi():
    response = test_client.get("/openapi.json")
    assert response.status_code == 200

"""
This test is currently failing with

>       return jsonable_encoder(OpenAPI(**output), by_alias=True, exclude_none=True)  # type: ignore
                                ^^^^^^^^^^^^^^^^^
E       pydantic_core._pydantic_core.ValidationError: 6 validation errors for OpenAPI
E       components.schemas.Simulation-Input.Schema.type.literal['array','boolean','integer','null','number','object','string']
E         Input should be 'array', 'boolean', 'integer', 'null', 'number', 'object' or 'string' [type=literal_error, input_value='simulation', input_type=str]
E           For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E       components.schemas.Simulation-Input.Schema.type.list[literal['array','boolean','integer','null','number','object','string']]
E         Input should be a valid list [type=list_type, input_value='simulation', input_type=str]
E           For further information visit https://errors.pydantic.dev/2.12/v/list_type
E       components.schemas.Simulation-Input.Reference.$ref
E         Field required [type=missing, input_value={'description': 'Record o...', 'type': 'simulation'}, input_type=dict]
E           For further information visit https://errors.pydantic.dev/2.12/v/missing
E       components.schemas.Simulation-Output.Schema.type.literal['array','boolean','integer','null','number','object','string']
E         Input should be 'array', 'boolean', 'integer', 'null', 'number', 'object' or 'string' [type=literal_error, input_value='simulation', input_type=str]
E           For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E       components.schemas.Simulation-Output.Schema.type.list[literal['array','boolean','integer','null','number','object','string']]
E         Input should be a valid list [type=list_type, input_value='simulation', input_type=str]
E           For further information visit https://errors.pydantic.dev/2.12/v/list_type
E       components.schemas.Simulation-Output.Reference.$ref
E         Field required [type=missing, input_value={'description': 'Record o...', 'type': 'simulation'}, input_type=dict]
E           For further information visit https://errors.pydantic.dev/2.12/v/missing

../env-prov-202602/lib/python3.11/site-packages/fastapi/openapi/utils.py:572: ValidationError

It appears to be because the "type" field is defined as a Literal with a single value,
seems to be related to
https://github.com/googleapis/python-genai/issues/264
and
https://github.com/fastapi/fastapi/discussions/11119

As a workaround, commenting-out the "type" field in simulation/examples.py removes the error

$ pip list
Package            Version
------------------ -----------
annotated-doc      0.0.4
annotated-types    0.7.0
anyio              4.12.1
asgi-matomo        0.6.1
asgiref            3.11.1
Authlib            1.6.7
certifi            2026.1.4
cffi               2.0.0
charset-normalizer 3.4.4
click              8.3.1
cryptography       46.0.5
ebrains_kg_core    0.9.20
fairgraph          0.13.2
fastapi            0.128.7
h11                0.16.0
httpcore           1.0.9
httpx              0.28.1
idna               3.11
iniconfig          2.3.0
itsdangerous       2.2.0
matomo-core        0.3.2
openMINDS          0.4.0
packaging          26.0
pip                26.0.1
pluggy             1.6.0
pycparser          3.0
pydantic           2.12.5
pydantic_core      2.41.5
Pygments           2.19.2
pytest             9.0.2
python-dateutil    2.9.0.post0
requests           2.32.5
rfc3987            1.3.8
setuptools         65.5.0
six                1.17.0
starlette          0.52.1
tabulate           0.9.0
typing_extensions  4.15.0
typing-inspection  0.4.2
urllib3            2.6.3
uvicorn            0.40.0
"""