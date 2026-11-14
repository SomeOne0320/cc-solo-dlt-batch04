import math
import dataclasses
import threading
from base64 import b64encode
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    cast,
)
from typing_extensions import Annotated
from requests.auth import AuthBase
from requests import PreparedRequest, Session as BaseSession  # noqa: I251

from dlt.common import logger
from dlt.common.exceptions import MissingDependencyException
from dlt.common.configuration.specs.base_configuration import configspec, NotResolved
from dlt.common.configuration.specs import CredentialsConfiguration
from dlt.common.configuration.specs.exceptions import NativeValueError
from dlt.common.pendulum import pendulum
from dlt.common.typing import TSecretStrValue
from dlt.sources.helpers.rest_client.exceptions import OAuth2TokenRefreshError

if TYPE_CHECKING:
    from _thread import LockType

    from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
else:
    PrivateKeyTypes = Any

TApiKeyLocation = Literal["header", "cookie", "query", "param"]  # Alias for scheme "in" field


class AuthConfigBase(AuthBase, CredentialsConfiguration):
    """Authenticator base which is both `requests` friendly AuthBase and dlt SPEC
    configurable via env variables or toml files
    """

    def __bool__(self) -> bool:
        # This is needed to avoid AuthConfigBase-derived classes
        # which do not implement CredentialsConfiguration interface
        # to be evaluated as False in requests.sessions.Session.prepare_request()
        return True


@configspec
class BearerTokenAuth(AuthConfigBase):
    """Uses `token` for Bearer authentication in "Authorization" header."""

    token: TSecretStrValue = None

    def parse_native_representation(self, value: Any) -> None:
        if isinstance(value, str):
            self.token = value
        else:
            raise NativeValueError(
                type(self),
                value,
                "`token` argument of `BearerTokenAuth` must be a `str`. Got value of type"
                f" `{type(value)}`",
            )

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


@configspec
class APIKeyAuth(AuthConfigBase):
    """Uses provided `api_key` to create authorization data in the specified `location` (query, param, header, cookie) under specified `name`"""

    name: str = "Authorization"
    api_key: TSecretStrValue = None
    location: TApiKeyLocation = "header"

    def parse_native_representation(self, value: Any) -> None:
        if isinstance(value, str):
            self.api_key = value
        else:
            raise NativeValueError(
                type(self),
                value,
                "`api_key` argument of `APIKeyAuth` must be a `str`. Got value of type"
                f" `{type(value)}`",
            )

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        if self.location == "header":
            request.headers[self.name] = self.api_key
        elif self.location in ["query", "param"]:
            request.prepare_url(request.url, {self.name: self.api_key})
        elif self.location == "cookie":
            raise NotImplementedError()
        return request


@configspec
class HttpBasicAuth(AuthConfigBase):
    """Uses HTTP basic authentication"""

    username: str = None
    password: TSecretStrValue = None

    def parse_native_representation(self, value: Any) -> None:
        if isinstance(value, Iterable) and not isinstance(value, str):
            value = list(value)
            if len(value) == 2:
                self.username, self.password = value
                return
        raise NativeValueError(
            type(self),
            value,
            "`username` and `password` arguments of `HttpBasicAuth` must be a `str` values. "
            f"Got values of type `{type(value)}`",
        )

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        encoded = b64encode(f"{self.username}:{self.password}".encode()).decode()
        request.headers["Authorization"] = f"Basic {encoded}"
        return request


@configspec
class OAuth2AuthBase(AuthConfigBase):
    """Base class for oauth2 authenticators. requires access_token"""

    access_token: TSecretStrValue = None

    def parse_native_representation(self, value: Any) -> None:
        if isinstance(value, str):
            self.access_token = value
        else:
            raise NativeValueError(
                type(self),
                value,
                "`access_token` argument of `OAuth2AuthBase` must be a `str`. Got value of type"
                f" `{type(value)}`",
            )

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        return request


@configspec
class OAuth2ClientCredentials(OAuth2AuthBase):
    """
    This class implements OAuth2 Client Credentials flow where the authorization service
    gives permission without the end user approving.
    This is often used for machine-to-machine authorization.
    The client sends its client ID and client secret to the authorization service which replies
    with a temporary access token.
    With the access token, the client can access resource services.
    """

    access_token: Annotated[Optional[TSecretStrValue], NotResolved()] = None
    access_token_url: str = None
    client_id: TSecretStrValue = None
    client_secret: TSecretStrValue = None
    access_token_request_data: Dict[str, Any] = None
    default_token_expiration: int = 3600
    session: Annotated[BaseSession, NotResolved()] = None

    def __post_init__(self) -> None:
        if self.access_token_request_data is None:
            self.access_token_request_data = {}
        self.token_expiry: pendulum.DateTime = pendulum.now()
        # use default system session unless specified otherwise
        if self.session is None:
            from dlt.sources.helpers import requests

            self.session = requests.client.session

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        if self.access_token is None or self.is_token_expired():
            self.obtain_token()
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        return request

    def is_token_expired(self) -> bool:
        return pendulum.now() >= self.token_expiry

    def obtain_token(self) -> None:
        response = self.session.post(self.access_token_url, **self.build_access_token_request())
        response.raise_for_status()
        response_json = response.json()
        self.parse_native_representation(self.parse_access_token(response_json))
        expires_in_seconds = self.parse_expiration_in_seconds(response_json)
        self.token_expiry = pendulum.now().add(seconds=expires_in_seconds)

    def build_access_token_request(self) -> Dict[str, Any]:
        return {
            "headers": {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            "data": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                **self.access_token_request_data,
            },
        }

    def parse_expiration_in_seconds(self, response_json: Any) -> int:
        return int(response_json.get("expires_in", self.default_token_expiration))

    def parse_access_token(self, response_json: Any) -> str:
        return str(response_json.get("access_token"))


@configspec
class OAuth2RefreshTokenAuth(OAuth2AuthBase):
    """OAuth 2.0 refresh token flow.

    Uses a long-lived `refresh_token` to obtain temporary access tokens. An access token is
    obtained on the first request and renewed whenever it expires. When the authorization
    server issues a new refresh token it replaces the old one, otherwise the old one is kept.
    """

    access_token: Annotated[Optional[TSecretStrValue], NotResolved()] = None
    """The current access token. Obtained automatically and never resolved from the config."""
    refresh_token: TSecretStrValue = None
    refresh_token_url: Optional[str] = None
    """Token endpoint exchanging the refresh token for an access token."""
    access_token_url: Optional[str] = None
    """Alias for `refresh_token_url`, matching `OAuth2ClientCredentials`."""
    refresh_token_request_data: Optional[Dict[str, Any]] = None
    """Additional data sent to the token endpoint as JSON."""
    access_token_request_data: Optional[Dict[str, Any]] = None
    """Alias for `refresh_token_request_data`, matching `OAuth2ClientCredentials`."""
    client_id: Optional[TSecretStrValue] = None
    client_secret: Optional[TSecretStrValue] = None
    default_token_expiration: int = 3600
    session: Annotated[BaseSession, NotResolved()] = None

    def __post_init__(self) -> None:
        self.token_expiry: Optional[pendulum.DateTime] = None
        # use default system session unless specified otherwise
        if self.session is None:
            from dlt.sources.helpers import requests

            self.session = requests.client.session

    def __call__(self, request: PreparedRequest) -> PreparedRequest:
        self.ensure_valid_access_token()
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        return request

    def token_url(self) -> Optional[str]:
        """Returns the configured token endpoint."""
        return self.refresh_token_url or self.access_token_url

    def is_token_expired(self) -> bool:
        if self.access_token is None:
            return True
        token_expiry = getattr(self, "token_expiry", None)
        return token_expiry is None or pendulum.now() >= token_expiry

    def ensure_valid_access_token(self) -> None:
        """Obtains a new access token if the current one is missing or expired.

        Concurrent callers sharing this authenticator refresh the token only once.
        """
        if not self.is_token_expired():
            return
        with self._get_refresh_lock():
            # another thread may have refreshed the token while we waited for the lock
            if not self.is_token_expired():
                return
            self.obtain_token()

    def _get_refresh_lock(self) -> "LockType":
        # the lock is not a config field, so copies (which skip `__post_init__`) create it lazily
        lock = self.__dict__.get("_refresh_lock")
        if lock is None:
            lock = self.__dict__.setdefault("_refresh_lock", threading.Lock())
        return lock

    def obtain_token(self) -> None:
        """Exchanges the refresh token for a new access token.

        Raises:
            OAuth2TokenRefreshError: If the refresh request fails or the response does not
                contain an access token. The previous authentication state is preserved, so
                the next request may retry the refresh.
        """
        token_url = self.token_url()
        if not self.refresh_token:
            raise OAuth2TokenRefreshError(
                "Cannot refresh the access token: `refresh_token` is not set"
            )
        if not token_url:
            raise OAuth2TokenRefreshError(
                "Cannot refresh the access token: neither `refresh_token_url` nor"
                " `access_token_url` is set"
            )

        response = None
        try:
            response = self.session.post(token_url, **self.build_refresh_token_request())
            response.raise_for_status()
            response_json = response.json()
            access_token = self.parse_access_token(response_json)
            new_refresh_token = self.parse_refresh_token(response_json)
            expires_in_seconds = self.parse_expiration_in_seconds(response_json)
        except Exception as ex:
            error = OAuth2TokenRefreshError(
                f"Could not refresh the access token using `{token_url}`: {ex}"
            )
            # `requests.Response` is falsy for error status codes, so compare with None
            error_response = getattr(ex, "response", None)
            if error_response is None:
                error_response = response
            if error_response is not None:
                error.response = error_response  # type: ignore[attr-defined]
            raise error from ex

        # update the state only after the response was validated, so a failed refresh
        # leaves the previous token intact and the next request can retry
        self.access_token = access_token
        if new_refresh_token:
            self.refresh_token = new_refresh_token
        self.token_expiry = pendulum.now().add(seconds=expires_in_seconds)

    def build_refresh_token_request(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        if self.client_id:
            data["client_id"] = self.client_id
        if self.client_secret:
            data["client_secret"] = self.client_secret
        extra_data = self.refresh_token_request_data or self.access_token_request_data
        if extra_data:
            data.update(extra_data)
        return {"json": data}

    def parse_expiration_in_seconds(self, response_json: Any) -> int:
        expires_in = response_json.get("expires_in")
        if expires_in is None:
            return self.default_token_expiration
        return int(expires_in)

    def parse_access_token(self, response_json: Any) -> str:
        access_token = response_json.get("access_token")
        if not access_token:
            raise ValueError("The token response does not contain `access_token`")
        return str(access_token)

    def parse_refresh_token(self, response_json: Any) -> Optional[str]:
        refresh_token = response_json.get("refresh_token")
        return str(refresh_token) if refresh_token else None

    def copy(self: "OAuth2RefreshTokenAuth") -> "OAuth2RefreshTokenAuth":
        # `BaseConfiguration.copy` copies dataclass fields only, keep the runtime token state
        new_auth = super().copy()
        new_auth.token_expiry = getattr(self, "token_expiry", None)
        return new_auth

    def __getstate__(self) -> Dict[str, Any]:
        # locks cannot be pickled or deep-copied; a copied object creates its own lock lazily
        state = self.__dict__.copy()
        state.pop("_refresh_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# aliases following the naming of the other OAuth 2.0 authenticators
OAuth2RefreshToken = OAuth2RefreshTokenAuth
RefreshTokenAuth = OAuth2RefreshTokenAuth


@configspec
class OAuthJWTAuth(BearerTokenAuth):
    """This is a form of Bearer auth, actually there's not standard way to declare it in openAPI"""

    format: Final[Literal["JWT"]] = dataclasses.field(  # noqa: A003
        default="JWT", init=False, repr=False, compare=False
    )
    client_id: str = None
    private_key: TSecretStrValue = None
    auth_endpoint: str = None
    scopes: Optional[Union[str, List[str]]] = None
    headers: Optional[Dict[str, str]] = None
    private_key_passphrase: Optional[TSecretStrValue] = None
    default_token_expiration: int = 3600
    session: Annotated[BaseSession, NotResolved()] = None

    def __post_init__(self) -> None:
        if self.scopes is not None and not isinstance(self.scopes, str):
            self.scopes = " ".join(self.scopes)
        self.token = None
        self.token_expiry: Optional[pendulum.DateTime] = None
        # use default system session unless specified otherwise
        if self.session is None:
            from dlt.sources.helpers import requests

            self.session = requests.client.session

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        if self.token is None or self.is_token_expired():
            self.obtain_token()
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r

    def is_token_expired(self) -> bool:
        return not self.token_expiry or pendulum.now() >= self.token_expiry

    def obtain_token(self) -> None:
        try:
            import jwt
        except ModuleNotFoundError:
            raise MissingDependencyException("dlt OAuth helpers", ["PyJWT"])

        payload = self.create_jwt_payload()
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt.encode(payload, self.load_private_key(), algorithm="RS256"),  # type: ignore[arg-type,unused-ignore]
        }

        logger.debug(f"Obtaining token from {self.auth_endpoint}")

        response = self.session.post(self.auth_endpoint, headers=self.headers, data=data)
        response.raise_for_status()

        token_response = response.json()
        self.token = token_response["access_token"]
        self.token_expiry = pendulum.now().add(
            seconds=token_response.get("expires_in", self.default_token_expiration)
        )

    def create_jwt_payload(self) -> Dict[str, Union[str, int]]:
        now = pendulum.now()
        payload: Dict[str, Union[str, int]] = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.auth_endpoint,
            "exp": math.floor((now.add(hours=1)).timestamp()),
            "iat": math.floor(now.timestamp()),
        }
        if self.scopes is not None:
            payload["scope"] = cast(str, self.scopes)
        return payload

    def load_private_key(self) -> "PrivateKeyTypes":
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
        except ModuleNotFoundError:
            raise MissingDependencyException("dlt OAuth helpers", ["cryptography"])

        private_key_bytes = self.private_key.encode("utf-8")
        return serialization.load_pem_private_key(
            private_key_bytes,
            password=(
                self.private_key_passphrase.encode("utf-8") if self.private_key_passphrase else None
            ),
            backend=default_backend(),
        )
