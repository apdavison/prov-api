import json
import os

HBP_IDENTITY_SERVICE_URL_V2 = "https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect"
HBP_COLLAB_SERVICE_URL_V2 = "https://wiki.ebrains.eu/rest/v1/"
EBRAINS_IAM_CONF_URL = "https://iam.ebrains.eu/auth/realms/hbp/.well-known/openid-configuration"
EBRAINS_IAM_CLIENT_ID = os.environ.get("EBRAINS_IAM_CLIENT_ID")
EBRAINS_IAM_SECRET = os.environ.get("EBRAINS_IAM_SECRET")
KG_SERVICE_ACCOUNT_CLIENT_ID = os.environ.get("KG_SERVICE_ACCOUNT_CLIENT_ID")
KG_SERVICE_ACCOUNT_SECRET = os.environ.get("KG_SERVICE_ACCOUNT_SECRET")
SESSIONS_SECRET_KEY = os.environ.get("SESSIONS_SECRET_KEY")
BASE_URL = os.environ.get("PROV_API_BASE_URL")
KG_CORE_API_HOST = os.environ.get("KG_CORE_API_HOST")
ADMIN_GROUP_ID = "computation-curators"

this_dir = os.path.dirname(__file__)
build_info_path = os.path.join(this_dir, "build_info.json")
if os.path.exists(build_info_path):
    with open(build_info_path, "r") as fp:
        BUILD_INFO = json.load(fp)
else:
    BUILD_INFO = None