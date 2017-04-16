# Base luigi Dockerfile with python3.5.2
FROM python:3.5.2
MAINTAINER Spiro Sideris <spirosideris at gmail.com>
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

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
RUN mkdir /etc/luigi && chmod -R a+rw /etc/luigi

COPY scripts/ .
RUN chmod +x generate_config.py && chmod +x run.sh && mv run.sh /usr/bin && chmod 777 Anaconda3-4.3.1-Linux-x86_64.sh && mv Anaconda3-4.3.1-Linux-x86_64.sh /usr/bin

USER luigi
RUN Anaconda3-4.3.1-Linux-x86_64.sh -b
ENV PATH=/home/luigi/anaconda3/bin:$PATH LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN pip install 'luigi' 'psycopg2'

VOLUME /home/luigi/love_matcher_project

CMD ["run.sh"]
