from typing import Any

from requests.exceptions import HTTPError

from dlt.common.exceptions import DltException


class RESTClientException(DltException):
    pass


class IgnoreResponseException(RESTClientException):
    pass


class PaginatorSetupError(RESTClientException, ValueError):
    pass


class PaginatorNotFound(RESTClientException):
    pass


class OAuth2TokenRefreshError(RESTClientException, HTTPError, ValueError):
    """Raised when an access token could not be refreshed.

    The authenticator keeps its previous state so the next request may retry the refresh.
    """

    def __reduce__(self) -> Any:
        # `DltException.__reduce__` bypasses exception construction and loses `args`,
        # which breaks the message when the exception is pickled
        return type(self), self.args, self.__dict__
