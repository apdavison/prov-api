import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import httpx

logger = logging.getLogger("matomo")


class MatomoMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, matomo_url: str, idsite: int):
        super().__init__(app)
        self.tracking_url = f"{matomo_url.rstrip('/')}/matomo.php"
        self.idsite = idsite

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        url = str(request.url)
        action_name = request.url.path

        params = {
            "idsite": self.idsite,
            "rec": 1,
            "url": url,
            "action_name": action_name,
            "send_image": 0,
        }

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
