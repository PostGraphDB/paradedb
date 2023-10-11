#!/bin/bash

# Exit on subcommand errors
set -Eeuo pipefail

if [ "$USE_CRUNCHY" = "true" ]; then
    apt install -y \
        pgbackrest-${BACKREST_VERSION} \
        postgresql-${PG_VERSION_MAJOR}-partman

    mkdir -p /opt/crunchy
    cp -R /tmp/crunchy/bin/postgres_common /opt/crunchy/bin/
    cp -R /tmp/crunchy/bin/common /opt/crunchy/bin/
    cp -R /tmp/crunchy/conf /opt/crunchy/conf/
    rm -rf /var/spool/pgbackrest
fi
