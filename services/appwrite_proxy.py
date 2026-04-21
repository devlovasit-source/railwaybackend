import os
import json
from typing import Any, Dict, List, Optional

import httpx


class AppwriteProxyError(Exception):
    pass


def _load_local_env() -> None:
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    candidate_paths = [
        os.path.join(cwd, ".env"),
        os.path.join(parent, ".env"),
        os.path.join(parent, "backend", ".env"),
        os.path.join(parent, "backend-master", ".env"),
        os.path.join(parent, "ahvi", ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            continue


class AppwriteProxy:
    def __init__(self) -> None:
        _load_local_env()
        self.endpoint = (
            os.getenv("APPWRITE_ENDPOINT", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_ENDPOINT", "")
        ).rstrip("/")
        self.project_id = (
            os.getenv("APPWRITE_PROJECT_ID", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_PROJECT_ID", "")
        )
        self.database_id = (
            os.getenv("APPWRITE_DATABASE_ID", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_DATABASE_ID", "")
        )
        self.api_key = (
            os.getenv("APPWRITE_API_KEY", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_API_KEY", "")
            or os.getenv("APPWRITE_KEY", "")
        )

        self.collection_map = {
            "outfits": os.getenv("APPWRITE_COLLECTION_OUTFITS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS", ""),
            "users": os.getenv("APPWRITE_COLLECTION_USERS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_USERS", ""),
            "plans": os.getenv("APPWRITE_COLLECTION_PLANS", "") or os.getenv("PLANS_COLLECTION_ID", ""),
            "saved_boards": os.getenv("APPWRITE_COLLECTION_SAVED_BOARDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SAVED_BOARDS", ""),
            "skincare": os.getenv("APPWRITE_COLLECTION_SKINCARE", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SKINCARE", ""),
            "skincare_profiles": (
                os.getenv("APPWRITE_COLLECTION_SKINCARE_PROFILES", "")
                or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SKINCARE_PROFILES", "")
                or os.getenv("APPWRITE_COLLECTION_SKINCARE", "")
                or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SKINCARE", "")
            ),
            "contacts": os.getenv("APPWRITE_COLLECTION_CONTACTS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_CONTACTS", ""),
            "workout_outfits": os.getenv("APPWRITE_COLLECTION_WORKOUT_OUTFITS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_WORKOUT_OUTFITS", ""),
            "bills": os.getenv("APPWRITE_COLLECTION_BILLS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_BILLS", ""),
            "coupons": os.getenv("APPWRITE_COLLECTION_COUPONS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_COUPONS", ""),
            "meds": os.getenv("APPWRITE_COLLECTION_MEDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MEDS", ""),
            "med_logs": os.getenv("APPWRITE_COLLECTION_MED_LOGS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MED_LOGS", ""),
            "logs": (
                os.getenv("APPWRITE_COLLECTION_LOGS", "")
                or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_LOGS", "")
                or os.getenv("APPWRITE_COLLECTION_MED_LOGS", "")
                or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MED_LOGS", "")
            ),
            "meal_plans": os.getenv("APPWRITE_COLLECTION_MEAL_PLANS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MEAL_PLANS", ""),
            "life_goals": os.getenv("APPWRITE_COLLECTION_LIFE_GOALS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_LIFE_GOALS", ""),
            "life_boards": os.getenv("APPWRITE_COLLECTION_LIFE_BOARDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_LIFE_BOARDS", ""),
            "chat_threads": os.getenv("APPWRITE_COLLECTION_CHAT_THREADS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_CHAT_THREADS", ""),
            "chat_messages": os.getenv("APPWRITE_COLLECTION_CHAT_MESSAGES", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_CHAT_MESSAGES", ""),
            "memories": os.getenv("APPWRITE_COLLECTION_MEMORIES", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MEMORIES", ""),
            "jobs": os.getenv("APPWRITE_COLLECTION_JOBS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_JOBS", ""),
        }
        self.resource_aliases = {
            "meal_planner": "meal_plans",
            "meal": "meal_plans",
            "medicines": "meds",
            "medicine": "meds",
            "log": "logs",
            "calendar": "plans",
            "workout": "workout_outfits",
            "workouts": "workout_outfits",
            "skincare": "skincare_profiles",
            "skin": "skincare_profiles",
            "skincare_profile": "skincare_profiles",
            "life_board": "life_boards",
            "lifeboard": "life_boards",
            "contacts": "users",
            "memory": "memories",
        }

        self.user_field_map = {
            "outfits": "userId",
            "users": None,
            "plans": "userId",
            "saved_boards": "userId",
            "skincare": "userId",
            "skincare_profiles": "userId",
            "contacts": "userId",
            "workout_outfits": "userId",
            "bills": "userId",
            "coupons": "userId",
            "meds": "userId",
            "med_logs": "userId",
            "logs": "userId",
            "meal_plans": "userId",
            "life_goals": "userId",
            "life_boards": "userId",
            "chat_threads": "userId",
            "chat_messages": "userId",
            "memories": "userId",
            "jobs": "userId",
        }

        self.order_query_map = {
            "outfits": {"method": "orderDesc", "attribute": "$createdAt"},
            "plans": None,
            "saved_boards": {"method": "orderDesc", "attribute": "$createdAt"},
            "skincare": None,
            "skincare_profiles": {"method": "orderDesc", "attribute": "$createdAt"},
            "contacts": {"method": "orderDesc", "attribute": "$createdAt"},
            "workout_outfits": {"method": "orderDesc", "attribute": "$createdAt"},
            "bills": {"method": "orderDesc", "attribute": "$createdAt"},
            "coupons": {"method": "orderDesc", "attribute": "$createdAt"},
            "meds": {"method": "orderDesc", "attribute": "$createdAt"},
            "med_logs": {"method": "orderDesc", "attribute": "time"},
            "logs": {"method": "orderDesc", "attribute": "takenAt"},
            "meal_plans": {"method": "orderDesc", "attribute": "$createdAt"},
            "life_goals": {"method": "orderDesc", "attribute": "$createdAt"},
            "life_boards": {"method": "orderDesc", "attribute": "$createdAt"},
            "chat_threads": {"method": "orderDesc", "attribute": "$updatedAt"},
            "chat_messages": {"method": "orderDesc", "attribute": "$createdAt"},
            "memories": {"method": "orderDesc", "attribute": "$updatedAt"},
            "jobs": {"method": "orderDesc", "attribute": "$createdAt"},
        }
        timeout_seconds = float(os.getenv("APPWRITE_TIMEOUT_SECONDS", "8"))
        self._http = httpx.Client(timeout=timeout_seconds)

    def _normalize_resource(self, resource: str) -> str:
        key = str(resource or "").strip()
        if not key:
            return key
        return self.resource_aliases.get(key, key)

    @staticmethod
    def _serialize_query_token(token: Any) -> str:
        if isinstance(token, dict):
            return json.dumps(token, separators=(",", ":"), ensure_ascii=False)
        return str(token)

    def _ensure_config(self) -> None:
        missing = []
        if not self.endpoint:
            missing.append("APPWRITE_ENDPOINT/EXPO_PUBLIC_APPWRITE_ENDPOINT")
        if not self.project_id:
            missing.append("APPWRITE_PROJECT_ID/EXPO_PUBLIC_APPWRITE_PROJECT_ID")
        if not self.database_id:
            missing.append("APPWRITE_DATABASE_ID/EXPO_PUBLIC_APPWRITE_DATABASE_ID")
        if not self.api_key:
            missing.append("APPWRITE_API_KEY/EXPO_PUBLIC_APPWRITE_API_KEY/APPWRITE_KEY")

        if missing:
            raise AppwriteProxyError(
                "Missing Appwrite backend configuration: " + ", ".join(missing)
            )

    def _collection_id(self, resource: str) -> str:
        resource = self._normalize_resource(resource)
        collection_id = self.collection_map.get(resource, "")
        if not collection_id:
            env_suffix = resource.upper()
            collection_id = (
                os.getenv(f"APPWRITE_COLLECTION_{env_suffix}", "")
                or os.getenv(f"EXPO_PUBLIC_APPWRITE_COLLECTION_{env_suffix}", "")
            )
        if not collection_id:
            collection_id = resource
        if not collection_id:
            raise AppwriteProxyError(f"Unknown or unconfigured resource: {resource}")
        return collection_id

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Appwrite-Project": self.project_id,
            "X-Appwrite-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _url(self, collection_id: str, document_id: Optional[str] = None) -> str:
        base = f"{self.endpoint}/databases/{self.database_id}/collections/{collection_id}/documents"
        if document_id:
            return f"{base}/{document_id}"
        return base

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_config()
        try:
            response = self._http.request(
                method=method,
                url=url,
                headers=self._headers(),
                params=params,
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise AppwriteProxyError(f"Appwrite connection failed: {exc}") from exc
        if response.status_code >= 400:
            raise AppwriteProxyError(
                f"Appwrite request failed ({response.status_code}): {response.text}"
            )
        if not response.text:
            return {}
        try:
            return response.json()
        except Exception as exc:
            raise AppwriteProxyError("Appwrite returned invalid JSON response.") from exc

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def _list_documents_page(
        self,
        collection_id: str,
        *,
        page_limit: int,
        offset: int,
        queries: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        query_tokens = list(queries or [])
        query_tokens.extend(
            [
                {"method": "limit", "values": [int(page_limit)]},
                {"method": "offset", "values": [int(offset)]},
            ]
        )

        serialized_tokens = [self._serialize_query_token(token) for token in query_tokens]

        indexed_queries: Dict[str, Any] = {}
        for idx, token in enumerate(serialized_tokens):
            indexed_queries[f"queries[{idx}]"] = token

        param_candidates: List[Dict[str, Any]] = [
            {"queries[]": serialized_tokens},
            {"queries": serialized_tokens},
            indexed_queries,
            {"limit": int(page_limit), "offset": int(offset)},
        ]

        for params in param_candidates:
            try:
                data = self._request(
                    "GET",
                    self._url(collection_id),
                    params=params,
                )
                docs = data.get("documents", [])
                if isinstance(docs, list):
                    return {
                        "documents": docs,
                        "total": data.get("total"),
                        "used_query_syntax": ("queries[]" in params or "queries" in params),
                    }
            except AppwriteProxyError:
                continue
            except Exception:
                continue

        if int(offset) == 0:
            data = self._request(
                "GET",
                self._url(collection_id),
            )
            docs = data.get("documents", [])
            if isinstance(docs, list):
                return {
                    "documents": docs,
                    "total": data.get("total"),
                    "used_query_syntax": False,
                }
        return {"documents": [], "total": None, "used_query_syntax": False}

    def _list_documents_local_filtered(
        self,
        collection_id: str,
        *,
        user_field: Optional[str],
        user_id: Optional[str],
        occasion: Optional[str],
        thread_id: Optional[str],
        request_id: Optional[str],
        status: Optional[str],
        type: Optional[str],
        order_query: Optional[Dict[str, Any]],
        limit: int,
        offset: int,
    ) -> Dict[str, Any]:
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        target_count = safe_offset + safe_limit + 1

        page_limit = min(100, max(25, safe_limit))
        raw_offset = 0
        max_scan = max(target_count * 5, 500)

        filtered_docs: List[Dict[str, Any]] = []
        seen_ids = set()

        while len(filtered_docs) < target_count and raw_offset < max_scan:
            page = self._list_documents_page(
                collection_id,
                page_limit=page_limit,
                offset=raw_offset,
                queries=[order_query] if order_query else [],
            )
            docs = page.get("documents", [])
            if not docs:
                break

            for d in docs:
                doc_id = str(d.get("$id", ""))
                if doc_id and doc_id in seen_ids:
                    continue
                if not self._matches_user(d, user_field, user_id):
                    continue
                if occasion and str(d.get("occasion", "")) != str(occasion):
                    continue
                if thread_id and str(d.get("threadId", "")) != str(thread_id):
                    continue
                if request_id and str(d.get("request_id", "")) != str(request_id):
                    continue
                if status and str(d.get("status", "")).strip().lower() != str(status).strip().lower():
                    continue
                if type and str(d.get("type", "")).strip().lower() != str(type).strip().lower():
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                filtered_docs.append(d)
                if len(filtered_docs) >= target_count:
                    break

            if len(docs) < page_limit:
                break
            raw_offset += page_limit

        page_docs = filtered_docs[safe_offset : safe_offset + safe_limit]
        has_more = len(filtered_docs) > (safe_offset + len(page_docs))
        return {
            "documents": page_docs,
            "total": None,
            "has_more": has_more,
            "next_offset": safe_offset + len(page_docs) if has_more else None,
        }

    @staticmethod
    def _equal_query(field: str, value: str) -> Dict[str, Any]:
        return {"method": "equal", "attribute": str(field), "values": [str(value)]}

    @staticmethod
    def _matches_user(doc: Dict[str, Any], user_field: Optional[str], user_id: Optional[str]) -> bool:
        if user_id is None or str(user_id) == "":
            return True
        expected = str(user_id)
        candidate_keys: List[str] = []
        if user_field:
            candidate_keys.append(user_field)
        for alias in ("userId", "userid", "user_id"):
            if alias not in candidate_keys:
                candidate_keys.append(alias)
        for key in candidate_keys:
            if str(doc.get(key, "")) == expected:
                return True
        return False

    def list_documents(
        self,
        resource: str,
        *,
        user_id: Optional[str] = None,
        occasion: Optional[str] = None,
        thread_id: Optional[str] = None,
        request_id: Optional[str] = None,
        status: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        return_meta: bool = False,
    ):
        resource = self._normalize_resource(resource)
        collection_id = self._collection_id(resource)
        user_field = self.user_field_map.get(resource)
        try:
            page_max = int(os.getenv("APPWRITE_PAGE_MAX", "100"))
        except Exception:
            page_max = 100
        safe_limit = max(1, min(int(limit), max(1, page_max)))
        safe_offset = max(0, int(offset))
        order_query = self.order_query_map.get(resource)

        query_tokens: List[Any] = []
        if order_query:
            query_tokens.append(order_query)
        if user_field and user_id:
            query_tokens.append(self._equal_query(user_field, str(user_id)))
        if occasion:
            query_tokens.append(self._equal_query("occasion", str(occasion)))
        if thread_id:
            query_tokens.append(self._equal_query("threadId", str(thread_id)))
        if request_id:
            query_tokens.append(self._equal_query("request_id", str(request_id)))
        if status:
            query_tokens.append(self._equal_query("status", str(status)))
        if type:
            query_tokens.append(self._equal_query("type", str(type)))

        page = self._list_documents_page(
            collection_id,
            page_limit=safe_limit,
            offset=safe_offset,
            queries=query_tokens,
        )
        docs = page.get("documents", [])
        total = page.get("total")
        used_query_syntax = bool(page.get("used_query_syntax"))

        filters_requested = bool((user_field and user_id) or occasion or thread_id or request_id or status or type)
        low_result_with_filter = filters_requested and safe_offset == 0 and len(docs) <= 1
        fallback_needed = ((not used_query_syntax) and (filters_requested or bool(order_query))) or low_result_with_filter

        if fallback_needed:
            fallback = self._list_documents_local_filtered(
                collection_id,
                user_field=user_field,
                user_id=user_id,
                occasion=occasion,
                thread_id=thread_id,
                request_id=request_id,
                status=status,
                type=type,
                order_query=order_query,
                limit=safe_limit,
                offset=safe_offset,
            )
            docs = fallback.get("documents", [])
            has_more = bool(fallback.get("has_more"))
            next_offset = fallback.get("next_offset")
            
            # --- DEBUG BLOCK FOR FALLBACK SCAN ---
            if resource == "outfits":
                print("\n=== WARDROBE FETCH DEBUG (FALLBACK MODE) ===")
                print(f"User ID Searched: {user_id}")
                print(f"Collection ID Used: {collection_id}")
                print(f"Total Items Found For You: {len(docs)}")
                if len(docs) > 0:
                    print(f"Sample Item Found: {docs[0].get('name')} - {docs[0].get('category')}")
                print("============================================\n")
            # -------------------------------------
            
            payload = {
                "documents": docs,
                "meta": {
                    "limit": safe_limit,
                    "offset": safe_offset,
                    "has_more": has_more,
                    "next_offset": next_offset,
                    "total": None,
                    "mode": "local_fallback",
                },
            }
            return payload if return_meta else docs

        has_more = False
        next_offset = None
        try:
            if total is not None:
                has_more = (safe_offset + len(docs)) < int(total)
            else:
                has_more = len(docs) >= safe_limit
        except Exception:
            has_more = len(docs) >= safe_limit

        if has_more:
            next_offset = safe_offset + len(docs)

        # --- DEBUG BLOCK FOR STANDARD QUERY ---
        if resource == "outfits":
            print("\n=== WARDROBE FETCH DEBUG (STANDARD QUERY) ===")
            print(f"User ID Searched: {user_id}")
            print(f"Collection ID Used: {collection_id}")
            print(f"Total Items Found For You: {len(docs)}")
            if len(docs) > 0:
                print(f"Sample Item Found: {docs[0].get('name')} - {docs[0].get('category')}")
            print("=============================================\n")
        # --------------------------------------

        payload = {
            "documents": docs,
            "meta": {
                "limit": safe_limit,
                "offset": safe_offset,
                "has_more": has_more,
                "next_offset": next_offset,
                "total": total,
                "mode": "query" if used_query_syntax else "plain_params",
            },
        }
        return payload if return_meta else docs

    def get_document(self, resource: str, document_id: str) -> Dict[str, Any]:
        resource = self._normalize_resource(resource)
        collection_id = self._collection_id(resource)
        return self._request("GET", self._url(collection_id, document_id))

    def create_document(self, resource: str, data: Dict[str, Any], document_id: str = "unique()") -> Dict[str, Any]:
        resource = self._normalize_resource(resource)
        collection_id = self._collection_id(resource)
        payload = {"documentId": document_id, "data": data}
        return self._request("POST", self._url(collection_id), payload=payload)

    def update_document(self, resource: str, document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        resource = self._normalize_resource(resource)
        collection_id = self._collection_id(resource)
        payload = {"data": data}
        return self._request("PATCH", self._url(collection_id, document_id), payload=payload)

    def delete_document(self, resource: str, document_id: str) -> None:
        resource = self._normalize_resource(resource)
        collection_id = self._collection_id(resource)
        self._request("DELETE", self._url(collection_id, document_id))
