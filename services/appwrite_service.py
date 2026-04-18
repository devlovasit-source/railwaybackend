import os

from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from dotenv import load_dotenv


def _env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

client = Client()
client.set_endpoint(
    _env_first(
        "APPWRITE_ENDPOINT",
        "EXPO_PUBLIC_APPWRITE_ENDPOINT",
        default="https://cloud.appwrite.io/v1",
    )
)
client.set_project(
    _env_first(
        "APPWRITE_PROJECT_ID",
        "APPWRITE_PROJECT",
        "EXPO_PUBLIC_APPWRITE_PROJECT_ID",
        default="69958f25003190519213",
    )
)

_appwrite_api_key = _env_first("APPWRITE_API_KEY", "APPWRITE_KEY")
if _appwrite_api_key:
    client.set_key(_appwrite_api_key)

account = Account(client)
databases = Databases(client)


def _build_client() -> Client:
    local_client = Client()
    local_client.set_endpoint(
        _env_first(
            "APPWRITE_ENDPOINT",
            "EXPO_PUBLIC_APPWRITE_ENDPOINT",
            default="https://cloud.appwrite.io/v1",
        )
    )
    local_client.set_project(
        _env_first(
            "APPWRITE_PROJECT_ID",
            "APPWRITE_PROJECT",
            "EXPO_PUBLIC_APPWRITE_PROJECT_ID",
            default="69958f25003190519213",
        )
    )
    if _appwrite_api_key:
        local_client.set_key(_appwrite_api_key)
    return local_client


def build_account_for_jwt(token: str) -> Account:
    local_client = _build_client()
    local_client.set_jwt(str(token or "").strip())
    return Account(local_client)
