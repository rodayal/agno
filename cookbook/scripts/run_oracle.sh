#!/bin/bash
set -euo pipefail

# One-command local Oracle Database for developing against agno's Oracle
# storage and vector database adapters. Uses gvenzl/oracle-free (community,
# no registry login or license click needed) rather than Oracle's own
# container-registry.oracle.com image, so a fresh clone works without any
# account setup. Both wrap the same Oracle Database Free (23ai) edition.
#
# Storage supports Oracle 19c and later; the vector database requires 23ai
# and later, which is what this script provides. See
# cookbook/06_storage/oracle/README.md for the version matrix and how
# db_schema and create_schema behave differently from Postgres on Oracle.

AGNO_ORACLE_CONTAINER="${AGNO_ORACLE_CONTAINER:-agno-oracle}"
AGNO_ORACLE_PORT="${AGNO_ORACLE_PORT:-1521}"

wait_for_oracle() {
  local attempt
  for ((attempt = 1; attempt <= 120; attempt++)); do
    if docker exec "$AGNO_ORACLE_CONTAINER" healthcheck.sh >/dev/null 2>&1; then
      return 0
    fi
    echo "  ...waiting for Oracle (${attempt}/120)"
    sleep 5
  done

  echo "Oracle did not become ready after 120 attempts." >&2
  docker logs --tail 50 "$AGNO_ORACLE_CONTAINER" >&2 || true
  return 1
}

echo "Starting Oracle Database Free container..."
docker run -d --name "$AGNO_ORACLE_CONTAINER" \
  -p "127.0.0.1:${AGNO_ORACLE_PORT}:1521" \
  -e ORACLE_PASSWORD=ai \
  -e APP_USER=ai \
  -e APP_USER_PASSWORD=ai \
  ghcr.io/gvenzl/oracle-free:latest

echo "Waiting for Oracle to become available (this can take a couple of minutes on first start)..."
wait_for_oracle
echo "Oracle is ready."

# Vector search needs an in-memory area sized for the HNSW graph. 256M is
# enough for the cookbook examples; raise it for a larger corpus.
echo "Configuring vector_memory_size for HNSW..."
docker exec -i "$AGNO_ORACLE_CONTAINER" sqlplus -s "sys/ai@localhost:1521/FREEPDB1 as sysdba" <<'SQL'
WHENEVER OSERROR EXIT 1
WHENEVER SQLERROR EXIT SQL.SQLCODE
ALTER SESSION SET CONTAINER = CDB$ROOT;
ALTER SYSTEM SET vector_memory_size = 256M SCOPE = SPFILE SID = '*';
EXIT SUCCESS
SQL
docker restart "$AGNO_ORACLE_CONTAINER" >/dev/null
echo "Waiting for Oracle to come back up after the restart..."
# Wait for the restarted database to pass its health check.
wait_for_oracle
echo "Configuring the FREEPDB1 vector-memory allowance..."
docker exec -i "$AGNO_ORACLE_CONTAINER" sqlplus -s "sys/ai@localhost:1521/FREEPDB1 as sysdba" <<'SQL'
WHENEVER OSERROR EXIT 1
WHENEVER SQLERROR EXIT SQL.SQLCODE
ALTER SYSTEM SET vector_memory_size = 256M SCOPE = BOTH;
SHOW PARAMETER vector_memory_size
EXIT SUCCESS
SQL
echo "Database setup complete!"

export ORACLE_HOST="localhost"
export ORACLE_PORT="$AGNO_ORACLE_PORT"
export ORACLE_DB_URL="oracle+oracledb://ai:ai@localhost:${AGNO_ORACLE_PORT}/?service_name=FREEPDB1"
export ORACLE_SERVICE_NAME="FREEPDB1"
export ORACLE_USER="ai"
export ORACLE_PASSWORD="ai"
