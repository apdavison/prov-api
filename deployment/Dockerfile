#
# Build an image for deploying the EBRAINS Provenance API
#
# To build the image, from the parent directory:
#   docker build -t prov_api -f deployment/Dockerfile .
#
# To run the application, the following environment variables must be provided:
#   - KG_SERVICE_ACCOUNT_CLIENT_ID
#   - KG_SERVICE_ACCOUNT_SECRET
#   - EBRAINS_IAM_CLIENT_ID
#   - EBRAINS_IAM_SECRET
#   - SESSIONS_SECRET_KEY
#   - PROV_API_BASE_URL
#   - KG_CORE_API_HOST

FROM docker-registry.ebrains.eu/prov-api/python:3.11

LABEL org.opencontainers.image.authors="Andrew Davison <andrew.davison@cnrs.fr>"

WORKDIR /code

COPY requirements.txt.lock /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY provenance /code/provenance
COPY deployment/build_info.json /code/provenance/build_info.json

RUN useradd appuser --uid 1001
USER appuser

CMD ["uvicorn", "provenance.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--timeout-keep-alive", "300"]
