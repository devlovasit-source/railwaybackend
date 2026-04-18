import os
from typing import Self

from pydantic import BaseModel, Field, model_validator


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class AppSettings(BaseModel):
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_window_seconds: int = Field(default=60)
    rate_limit_max_requests: int = Field(default=120)
    rate_limit_require_redis: bool = Field(default=False)
    rate_limit_fail_closed: bool = Field(default=False)
    try_on_daily_limit: int = Field(default=2)
    upload_max_bytes: int = Field(default=5 * 1024 * 1024)
    auth_cache_ttl_seconds: int = Field(default=30)
    auth_required: bool = Field(default=False)
    redis_url: str = Field(default="redis://localhost:6379/0")

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.rate_limit_window_seconds <= 0:
            self.rate_limit_window_seconds = 60
        if self.rate_limit_max_requests <= 0:
            self.rate_limit_max_requests = 120
        if self.upload_max_bytes <= 0:
            self.upload_max_bytes = 5 * 1024 * 1024
        if self.auth_cache_ttl_seconds <= 0:
            self.auth_cache_ttl_seconds = 30
        if self.try_on_daily_limit <= 0:
            self.try_on_daily_limit = 2
        return self

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            rate_limit_enabled=_env_bool("RATE_LIMIT_ENABLED", True),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            rate_limit_max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "120")),
            rate_limit_require_redis=_env_bool("RATE_LIMIT_REQUIRE_REDIS", False),
            rate_limit_fail_closed=_env_bool("RATE_LIMIT_FAIL_CLOSED", False),
            try_on_daily_limit=int(os.getenv("TRY_ON_DAILY_LIMIT", "2")),
            upload_max_bytes=int(os.getenv("UPLOAD_MAX_BYTES", str(5 * 1024 * 1024))),
            auth_cache_ttl_seconds=int(os.getenv("AUTH_CACHE_TTL_SECONDS", "30")),
            auth_required=_env_bool("AUTH_REQUIRED", False),
            redis_url=str(os.getenv("REDIS_URL", "redis://localhost:6379/0")),
        )


settings = AppSettings.from_env()
