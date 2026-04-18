import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from services.image_fingerprint import hamming_distance_hex

# Load env variables
load_dotenv()


class QdrantService:

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.image_collection = os.getenv("QDRANT_IMAGE_COLLECTION", "wardrobe_image")
        self.vector_size = 384
        self.memory_vector_size = 8
        try:
            self.image_vector_size = int(os.getenv("QDRANT_IMAGE_VECTOR_SIZE", "512"))
        except Exception:
            self.image_vector_size = 512
        self._initialized = False
        self.client = None
        self._vector_name_cache = {}
        self._vector_dim_cache = {}

        if self.url:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            except Exception as e:
                print("Qdrant client init failed:", str(e))
                self.client = None

    def enabled(self) -> bool:
        return self.client is not None

    def _ensure_ready(self) -> bool:
        if not self.enabled():
            return False
        if not self._initialized:
            self.init()
        return True

    # -------------------------
    # INIT (SAFE - CALL ON STARTUP)
    # -------------------------
    def init(self):
        if not self.enabled():
            print("Qdrant disabled: missing QDRANT_URL or client init failed")
            return

        if self._initialized:
            return

        print("Initializing Qdrant...")
        self._ensure_collection()
        self._ensure_memory_collection()
        self._ensure_image_collection()
        self._refresh_vector_name_cache()
        self._initialized = True

    # -------------------------
    # CREATE COLLECTIONS
    # -------------------------
    def _ensure_collection(self):
        if not self.enabled():
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print("Qdrant init error:", str(e))

    def _ensure_memory_collection(self):
        if not self.enabled():
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.memory_collection not in names:
                self.client.create_collection(
                    collection_name=self.memory_collection,
                    vectors_config=VectorParams(size=self.memory_vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print("Memory collection init error:", str(e))

    def _ensure_image_collection(self):
        if not self.enabled():
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.image_collection not in names:
                self.client.create_collection(
                    collection_name=self.image_collection,
                    vectors_config=VectorParams(size=self.image_vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print("Image collection init error:", str(e))

    def _refresh_vector_name_cache(self):
        self._vector_name_cache = {
            self.collection: self._detect_vector_name(self.collection),
            self.memory_collection: self._detect_vector_name(self.memory_collection),
            self.image_collection: self._detect_vector_name(self.image_collection),
        }
        self._vector_dim_cache = {
            self.collection: self._detect_vector_dim(self.collection),
            self.memory_collection: self._detect_vector_dim(self.memory_collection),
            self.image_collection: self._detect_vector_dim(self.image_collection),
        }

    def _detect_vector_name(self, collection_name: str):
        if not self.enabled():
            return None
        try:
            info = self.client.get_collection(collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)

            # Unnamed single-vector collection.
            if isinstance(vectors, VectorParams):
                return None

            # Named-vector collection.
            if isinstance(vectors, dict) and vectors:
                return next(iter(vectors.keys()))
        except Exception:
            return None
        return None

    def _detect_vector_dim(self, collection_name: str):
        if not self.enabled():
            return None
        try:
            info = self.client.get_collection(collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)
            name = self._vector_name_cache.get(collection_name)

            if isinstance(vectors, VectorParams):
                return int(vectors.size)

            if isinstance(vectors, dict) and vectors:
                if name and name in vectors:
                    cfg = vectors[name]
                else:
                    cfg = next(iter(vectors.values()))
                size = getattr(cfg, "size", None)
                if size is not None:
                    return int(size)
        except Exception:
            return None
        return None

    def _adapt_vector_dim(self, collection_name: str, vector: list):
        expected = self._vector_dim_cache.get(collection_name)
        if not expected:
            return vector
        current = len(vector or [])
        if current == expected:
            return vector
        if current > expected:
            return vector[:expected]
        return (vector or []) + ([0.0] * (expected - current))

    def _point_vector(self, collection_name: str, vector: list):
        vector = self._adapt_vector_dim(collection_name, vector)
        name = self._vector_name_cache.get(collection_name)
        if name:
            return {name: vector}
        return vector

    def _query_vector(self, collection_name: str, vector: list):
        vector = self._adapt_vector_dim(collection_name, vector)
        name = self._vector_name_cache.get(collection_name)
        if name:
            return (name, vector)
        return vector

    def _query_points_query(self, collection_name: str, vector: list):
        vector = self._adapt_vector_dim(collection_name, vector)
        return vector

    def _query_points_using(self, collection_name: str):
        return self._vector_name_cache.get(collection_name)

    def _extract_points_from_query_response(self, response):
        if response is None:
            return []
        if isinstance(response, list):
            return response
        if hasattr(response, "points") and getattr(response, "points") is not None:
            return response.points
        if isinstance(response, tuple) and len(response) >= 1:
            first = response[0]
            if isinstance(first, list):
                return first
        return []

    def _extract_points_from_scroll_response(self, response):
        if response is None:
            return [], None

        if isinstance(response, tuple):
            points = []
            next_offset = None
            if len(response) >= 1:
                first = response[0]
                if isinstance(first, list):
                    points = first
                elif hasattr(first, "points"):
                    points = getattr(first, "points") or []
            if len(response) >= 2:
                next_offset = response[1]
            return points, next_offset

        points = []
        if hasattr(response, "points"):
            points = getattr(response, "points") or []
        next_offset = getattr(response, "next_page_offset", None)
        return points, next_offset

    @staticmethod
    def _client_side_filter_points(points, user_id: str, limit: int):
        filtered = []
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            if str(payload.get("userId", "")) != str(user_id):
                continue
            if str(payload.get("feedback", "")) == "down":
                continue
            filtered.append(p)
            if len(filtered) >= limit:
                break
        return filtered

    def _scroll_user_points(self, user_id: str, scan_limit: int = 512, page_size: int = 128):
        if not self._ensure_ready():
            return []

        scan_limit = max(1, int(scan_limit))
        page_size = max(1, min(int(page_size), scan_limit))
        query_filter = {"must": [{"key": "userId", "match": {"value": user_id}}]}

        collected = []
        next_offset = None

        while len(collected) < scan_limit:
            batch_size = min(page_size, scan_limit - len(collected))
            kwargs = {
                "collection_name": self.collection,
                "limit": batch_size,
                "with_payload": True,
                "with_vectors": False,
            }
            if next_offset is not None:
                kwargs["offset"] = next_offset

            response = None
            used_server_filter = True
            try:
                response = self.client.scroll(**kwargs, scroll_filter=query_filter)
            except TypeError:
                try:
                    response = self.client.scroll(**kwargs, query_filter=query_filter)
                except TypeError:
                    try:
                        response = self.client.scroll(**kwargs, filter=query_filter)
                    except Exception:
                        response = None
                except Exception:
                    response = None
            except Exception:
                response = None

            if response is None:
                used_server_filter = False
                try:
                    response = self.client.scroll(**kwargs)
                except Exception:
                    return collected

            points, next_offset = self._extract_points_from_scroll_response(response)
            if not points:
                break

            if not used_server_filter:
                points = self._client_side_filter_points(points, user_id, batch_size)
                if not points and next_offset is None:
                    break

            collected.extend(points)
            if next_offset is None:
                break

        return collected

    def _client_search(self, collection_name: str, vector: list, user_id: str, limit: int):
        query_filter = {
            "must": [{"key": "userId", "match": {"value": user_id}}],
            "must_not": [{"key": "feedback", "match": {"value": "down"}}],
        }
        query_vector = self._query_vector(collection_name, vector)
        query_points_query = self._query_points_query(collection_name, vector)

        # qdrant-client (older): .search()
        if hasattr(self.client, "search"):
            try:
                return self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
            except Exception:
                # Some Qdrant deployments require a payload index for filtered search.
                # Fallback: nearest-neighbors without server-side filter, then filter client-side.
                points = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=max(limit * 20, 50),
                )
                return self._client_side_filter_points(points, user_id, limit)

        # qdrant-client (newer): .query_points()
        if hasattr(self.client, "query_points"):
            try:
                using_name = self._query_points_using(collection_name)
                kwargs = {
                    "collection_name": collection_name,
                    "query": query_points_query,
                    "limit": limit,
                    "query_filter": query_filter,
                }
                if using_name:
                    kwargs["using"] = using_name
                response = self.client.query_points(**kwargs)
                return self._extract_points_from_query_response(response)
            except TypeError:
                # Some versions expect query_vector keyword.
                try:
                    response = self.client.query_points(
                        collection_name=collection_name,
                        query_vector=query_points_query,
                        limit=limit,
                        query_filter=query_filter,
                    )
                    return self._extract_points_from_query_response(response)
                except Exception:
                    try:
                        # Fallback when `using` is not supported by client version.
                        response = self.client.query_points(
                            collection_name=collection_name,
                            query=query_points_query,
                            limit=limit,
                            query_filter=query_filter,
                        )
                        return self._extract_points_from_query_response(response)
                    except Exception:
                        pass
            except Exception:
                pass

            # Compatibility fallback: fetch nearest points without server-side filter
            # and filter client-side.
            try:
                using_name = self._query_points_using(collection_name)
                kwargs = {
                    "collection_name": collection_name,
                    "query": query_points_query,
                    "limit": max(limit * 20, 50),
                }
                if using_name:
                    kwargs["using"] = using_name
                response = self.client.query_points(**kwargs)
            except TypeError:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query_vector=query_points_query,
                    limit=max(limit * 20, 50),
                )
            except Exception:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_points_query,
                    limit=max(limit * 20, 50),
                )
            points = self._extract_points_from_query_response(response)
            return self._client_side_filter_points(points, user_id, limit)

        raise AttributeError("Qdrant client has neither 'search' nor 'query_points'")

    # -------------------------
    # UPSERT
    # -------------------------
    def upsert_item(self, item_id: str, vector: list, payload: dict):
        if not self._ensure_ready():
            return
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=item_id, vector=self._point_vector(self.collection, vector), payload=payload)],
            )
        except Exception as e:
            print("Upsert failed:", str(e))

    def upsert_memory_vector(self, point_id: str, vector: list, payload: dict):
        if not self._ensure_ready():
            return
        try:
            self.client.upsert(
                collection_name=self.memory_collection,
                points=[PointStruct(id=point_id, vector=self._point_vector(self.memory_collection, vector), payload=payload)],
            )
        except Exception as e:
            print("Memory upsert failed:", str(e))

    def upsert_image_vector(self, point_id: str, vector: list, payload: dict):
        if not self._ensure_ready():
            return
        if not vector:
            return
        try:
            self.client.upsert(
                collection_name=self.image_collection,
                points=[PointStruct(id=point_id, vector=self._point_vector(self.image_collection, vector), payload=payload)],
            )
        except Exception as e:
            print("Image vector upsert failed:", str(e))

    def _delete_point_from_collection(self, collection_name: str, item_id: str):
        if not item_id:
            return
        try:
            if hasattr(self.client, "delete"):
                try:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=[item_id],
                    )
                    return
                except Exception:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector={"points": [item_id]},
                    )
                    return
            self.client.delete_points(
                collection_name=collection_name,
                points=[item_id],
            )
        except Exception as e:
            print(f"Delete from Qdrant failed ({collection_name}):", str(e))

    def delete_item(self, item_id: str):
        if not self._ensure_ready():
            return
        self._delete_point_from_collection(self.collection, item_id)
        self._delete_point_from_collection(self.image_collection, item_id)

    # -------------------------
    # SEARCH
    # -------------------------
    def search_similar(self, vector: list, user_id: str, limit: int = 5):
        if not self._ensure_ready():
            return []
        try:
            results = self._client_search(self.collection, vector, user_id, limit)

            return [
                {"id": str(r.id), "score": self._boost_score(float(r.score), r.payload), "payload": r.payload or {}}
                for r in results
            ]

        except Exception as e:
            print("Search failed:", str(e))
            return []

    def search_similar_image(self, vector: list, user_id: str, limit: int = 5):
        if not self._ensure_ready():
            return []
        if not vector:
            return []
        try:
            results = self._client_search(self.image_collection, vector, user_id, limit)
            return [
                {"id": str(r.id), "score": float(r.score), "payload": r.payload or {}}
                for r in results
            ]
        except Exception as e:
            print("Image search failed:", str(e))
            return []

    def semantic_retrieve(self, vector: list, user_id: str, limit: int = 40):
        if not self._ensure_ready():
            return []
        try:
            query_filter = {"must": [{"key": "userId", "match": {"value": user_id}}]}
            query_vector = self._query_vector(self.collection, vector)
            query_points_query = self._query_points_query(self.collection, vector)

            if hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
            elif hasattr(self.client, "query_points"):
                try:
                    using_name = self._query_points_using(self.collection)
                    kwargs = {
                        "collection_name": self.collection,
                        "query": query_points_query,
                        "limit": limit,
                        "query_filter": query_filter,
                    }
                    if using_name:
                        kwargs["using"] = using_name
                    response = self.client.query_points(**kwargs)
                except TypeError:
                    response = self.client.query_points(
                        collection_name=self.collection,
                        query_vector=query_points_query,
                        limit=limit,
                        query_filter=query_filter,
                    )
                except Exception:
                    response = self.client.query_points(
                        collection_name=self.collection,
                        query=query_points_query,
                        limit=limit,
                        query_filter=query_filter,
                    )
                results = self._extract_points_from_query_response(response)
            else:
                raise AttributeError("Qdrant client has neither 'search' nor 'query_points'")

            return [
                {"id": str(r.id), "score": float(r.score), "payload": r.payload or {}}
                for r in results
            ]
        except Exception as e:
            print("Semantic retrieve failed:", str(e))
            return []

    # -------------------------
    # FEEDBACK UPDATE
    # -------------------------
    def update_feedback(self, item_id: str, feedback: str):
        if not self._ensure_ready():
            return
        try:
            self.client.set_payload(
                collection_name=self.collection,
                payload={"feedback": feedback},
                points=[item_id],
            )
        except Exception as e:
            print("Feedback update failed:", str(e))

    # -------------------------
    # BOOST SCORES
    # -------------------------
    def _boost_score(self, score: float, payload: dict):
        if not payload:
            return score

        feedback = payload.get("feedback")

        if feedback == "up":
            return score + 0.05

        if feedback == "down":
            return score - 0.2

        return score

    # -------------------------
    # DUPLICATE CHECK
    # -------------------------
    def find_image_duplicate(self, vector: list, user_id: str, threshold: float = 0.985):
        result = {
            "checked": False,
            "is_duplicate": False,
            "id": None,
            "score": 0.0,
            "threshold": float(threshold),
            "payload": {},
        }
        if not self._ensure_ready():
            return result
        if not vector:
            return result

        result["checked"] = True
        try:
            points = self._client_search(self.image_collection, vector, user_id, limit=1)
            if not points:
                return result

            top = points[0]
            score = float(getattr(top, "score", 0.0))
            payload = getattr(top, "payload", {}) or {}
            point_id = str(getattr(top, "id", "") or "")
            result.update(
                {
                    "is_duplicate": score >= float(threshold),
                    "id": point_id,
                    "score": score,
                    "payload": payload,
                }
            )
        except Exception as e:
            print("Image duplicate check failed:", str(e))
        return result

    def find_pixel_duplicate(
        self,
        user_id: str,
        pixel_hash: str,
        max_distance: int = 6,
        scan_limit: int = 512,
    ):
        result = {
            "checked": False,
            "is_duplicate": False,
            "id": None,
            "distance": None,
            "max_distance": int(max_distance),
            "pixel_hash": str(pixel_hash or "").strip().lower(),
            "payload": {},
        }
        if not self._ensure_ready():
            return result
        if not result["pixel_hash"]:
            return result

        result["checked"] = True
        best = None

        try:
            points = self._scroll_user_points(user_id, scan_limit=scan_limit)
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                if str(payload.get("feedback", "")) == "down":
                    continue
                candidate_hash = str(payload.get("pixel_hash") or "").strip().lower()
                if not candidate_hash:
                    continue

                distance = hamming_distance_hex(result["pixel_hash"], candidate_hash)
                if distance is None:
                    continue

                if best is None or distance < best["distance"]:
                    best = {
                        "id": str(getattr(point, "id", "") or ""),
                        "distance": int(distance),
                        "payload": payload,
                    }

                if int(distance) <= int(max_distance):
                    result.update(
                        {
                            "is_duplicate": True,
                            "id": str(getattr(point, "id", "") or ""),
                            "distance": int(distance),
                            "payload": payload,
                        }
                    )
                    return result

            if best is not None:
                result.update(
                    {
                        "id": best["id"],
                        "distance": best["distance"],
                        "payload": best["payload"],
                    }
                )
        except Exception as e:
            print("Pixel duplicate check failed:", str(e))

        return result

    def find_duplicate(self, vector: list, user_id: str, threshold: float = 0.97):
        result = {
            "is_duplicate": False,
            "id": None,
            "score": 0.0,
            "threshold": float(threshold),
            "payload": {},
        }
        if not self._ensure_ready():
            return result
        try:
            points = self._client_search(self.collection, vector, user_id, limit=1)
            if not points:
                return result

            top = points[0]
            score = float(getattr(top, "score", 0.0))
            payload = getattr(top, "payload", {}) or {}
            point_id = str(getattr(top, "id", "") or "")

            result.update(
                {
                    "id": point_id,
                    "score": score,
                    "payload": payload,
                    "is_duplicate": score >= float(threshold),
                }
            )
        except Exception as e:
            print("Duplicate check failed:", str(e))
        return result

    def is_duplicate(self, vector: list, user_id: str, threshold: float = 0.97):
        duplicate = self.find_duplicate(vector, user_id, threshold=threshold)
        return bool(duplicate.get("is_duplicate"))

    def status(self) -> dict:
        if not self.enabled():
            return {"enabled": False, "initialized": False, "url_configured": bool(self.url)}

        details = {
            "enabled": True,
            "initialized": self._initialized,
            "url_configured": bool(self.url),
            "collection": self.collection,
            "memory_collection": self.memory_collection,
            "image_collection": self.image_collection,
        }

        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            details["collections"] = names
            details["collection_ready"] = self.collection in names
            details["memory_collection_ready"] = self.memory_collection in names
            details["image_collection_ready"] = self.image_collection in names
        except Exception as e:
            details["error"] = str(e)

        return details


# -------------------------
# SINGLETON INSTANCE
# -------------------------
qdrant_service = QdrantService()
