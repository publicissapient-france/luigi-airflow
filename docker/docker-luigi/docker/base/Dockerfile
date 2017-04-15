# Base luigi Dockerfile with python3.5.2
FROM python:3.5.2
MAINTAINER Spiro Sideris <spirosideris at gmail.com>

# Run an update, create a luigi user with sudo permissions, and install
# python tools for luigi.
RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -q update && \
    useradd --password NP --create-home --shell /bin/bash luigi && \
    apt-get install -q -y --no-install-recommends \
        sudo cron python-dev libpq-dev build-essential \
        postgresql-client \
    && \
    echo "luigi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the luigi package and the database requirements for postgres task
# history.
RUN pip install 'luigi'
RUN mkdir /etc/luigi && chmod -R a+rw /etc/luigi

COPY scripts/ .
ADD https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh .
RUN chmod +x generate_config.py && chmod +x run.sh && mv run.sh /usr/bin && chmod 777 Anaconda2-4.3.1-Linux-x86_64.sh && mv Anaconda2-4.3.1-Linux-x86_64.sh /usr/bin

USER luigi

RUN Anaconda2-4.3.1-Linux-x86_64.sh -b

CMD ["run.sh"]
