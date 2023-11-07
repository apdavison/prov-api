#
# Build an image for deploying the EBRAINS Provenance API
#
# To build the image, from the parent directory:
#   docker build -t prov_api -f deployment/Dockerfile .
#
# To run the application:
#   docker run -d -p 443:443 -v /etc/letsencrypt:/etc/letsencrypt \
#              -e KG_SERVICE_ACCOUNT_REFRESH_TOKEN \
#              -e KG_SERVICE_ACCOUNT_CLIENT_ID \
#              -e KG_SERVICE_ACCOUNT_SECRET \
#              prov_api

FROM debian:bullseye-slim

MAINTAINER Andrew Davison <andrew.davison@cnrs.fr>

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --fix-missing; apt-get -y -q install python-pkg-resources python3-pkg-resources python3-pip supervisor build-essential nginx-extras git wget
RUN unset DEBIAN_FRONTEND

RUN pip3 install --upgrade pip

ADD https://api.github.com/repos/HumanBrainProject/fairgraph/git/refs/heads/master fairgraph_version.json
RUN git clone --branch master https://github.com/HumanBrainProject/fairgraph.git
#RUN pip3 install -r fairgraph/requirements.txt
RUN pip3 install ./fairgraph

ENV SITEDIR /home/docker/site

COPY requirements.txt $SITEDIR/
RUN pip3 install -r $SITEDIR/requirements.txt

COPY provenance $SITEDIR/provenance

ENV PYTHONPATH  /home/docker:/home/docker/site:/usr/lib/python2.7/dist-packages/:/usr/local/lib/python3.7/dist-packages:/usr/lib/python3.7/dist-packages

RUN echo "daemon off;" >> /etc/nginx/nginx.conf
RUN rm /etc/nginx/sites-enabled/default
COPY deployment/nginx-app.conf /etc/nginx/sites-enabled/
COPY deployment/supervisor-app.conf /etc/supervisor/conf.d/

EXPOSE 443
#EXPOSE 80

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisor-app.conf"]
