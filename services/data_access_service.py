from typing import Any, Dict

from services.appwrite_proxy import AppwriteProxy


def create_document(*, resource: str, payload: Dict[str, Any], document_id: str = "unique()"):
    return AppwriteProxy().create_document(resource, payload, document_id=document_id)


def update_document(*, resource: str, document_id: str, payload: Dict[str, Any]):
    return AppwriteProxy().update_document(resource, document_id, payload)


def delete_document(*, resource: str, document_id: str):
    AppwriteProxy().delete_document(resource, document_id)


def upsert_user_profile(*, user_id: str, payload: Dict[str, Any]):
    proxy = AppwriteProxy()
    try:
        return proxy.update_document("users", user_id, payload)
    except Exception:
        return proxy.create_document("users", payload, document_id=user_id)
