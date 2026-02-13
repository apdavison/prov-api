import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import httpx

logger = logging.getLogger("matomo")


def _extract_visitor_id(auth_header, secret):
    """Derive a 16-char hex visitor ID from the JWT's preferred_username."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    parts = token.split(".")
    if len(parts) != 3:
        logger.warning("JWT does not have 3 segments")
        return None
    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
    except binascii.Error as err:
        logger.warning("JWT payload is not valid base64: %s", err)
        return None
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as err:
        logger.warning("JWT payload is not valid JSON: %s", err)
        return None
    username = payload.get("preferred_username")
    if not username:
        return None
    return hmac.new(
        secret.encode(), username.encode(), hashlib.sha256
    ).hexdigest()[:16]


class MatomoMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, matomo_url: str, idsite: int, secret: str, token_auth: str = None):
        super().__init__(app)
        self.tracking_url = f"{matomo_url.rstrip('/')}/matomo.php"
        self.idsite = idsite
        self.secret = secret
        self.token_auth = token_auth

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        if response.status_code == 404:
            return response
        server_time_ms = int((time.monotonic() - start) * 1000)

        url = str(request.url)
        action_name = request.url.path

        params = {
            "idsite": self.idsite,
            "rec": 1,
            "url": url,
            "action_name": action_name,
            "send_image": 0,
            "pf_srv": server_time_ms,
        }
        ua = request.headers.get("user-agent")
        if ua:
            params["ua"] = ua
        lang = request.headers.get("accept-language")
        if lang:
            params["lang"] = lang
        urlref = request.headers.get("referer")
        if urlref:
            params["urlref"] = urlref
        visitor_id = _extract_visitor_id(
            request.headers.get("authorization"), self.secret
        )
        if visitor_id:
            params["_id"] = visitor_id
        if self.token_auth:
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            else:
                client_ip = request.client.host if request.client else None
            if client_ip:
                params["cip"] = client_ip
                params["token_auth"] = self.token_auth

        try:
            async with httpx.AsyncClient() as client:
                tracking_response = await client.post(self.tracking_url, params=params)
            logger.info(
                "Matomo tracking POST to %s with params %s returned status %s: %s",
                self.tracking_url,
                params,
                tracking_response.status_code,
                tracking_response.text[:200],
            )
        except Exception as exc:
            logger.error("Matomo tracking request failed: %s", exc)

        return response
