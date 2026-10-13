@echo off

if not defined AGNO_ORACLE_CONTAINER set "AGNO_ORACLE_CONTAINER=agno-oracle"
if not defined AGNO_ORACLE_PORT set "AGNO_ORACLE_PORT=1521"

echo Starting Oracle Database Free container...
docker run -d ^
  --name "%AGNO_ORACLE_CONTAINER%" ^
  -e ORACLE_PASSWORD=ai ^
  -e APP_USER=ai ^
  -e APP_USER_PASSWORD=ai ^
  -p "127.0.0.1:%AGNO_ORACLE_PORT%:1521" ^
  ghcr.io/gvenzl/oracle-free:latest

if errorlevel 1 (
  echo Failed to create the Oracle container.
  exit /b 1
)

echo Waiting for Oracle to become available...
call :wait_for_oracle
if errorlevel 1 exit /b 1

echo Configuring the root vector-memory pool...
(
  echo WHENEVER OSERROR EXIT 1
  echo WHENEVER SQLERROR EXIT SQL.SQLCODE
  echo ALTER SESSION SET CONTAINER = CDB$ROOT;
  echo ALTER SYSTEM SET vector_memory_size = 256M SCOPE = SPFILE SID = '*';
  echo EXIT SUCCESS
) | docker exec -i "%AGNO_ORACLE_CONTAINER%" sqlplus -s "sys/ai@localhost:1521/FREEPDB1 as sysdba"

if errorlevel 1 (
  echo Failed to configure the root vector-memory pool.
  exit /b 1
)

docker restart "%AGNO_ORACLE_CONTAINER%"
if errorlevel 1 (
  echo Failed to restart the Oracle container.
  exit /b 1
)

echo Waiting for Oracle after restart...
call :wait_for_oracle
if errorlevel 1 exit /b 1

echo Configuring the FREEPDB1 vector-memory allowance...
(
  echo WHENEVER OSERROR EXIT 1
  echo WHENEVER SQLERROR EXIT SQL.SQLCODE
  echo ALTER SYSTEM SET vector_memory_size = 256M SCOPE = BOTH;
  echo SHOW PARAMETER vector_memory_size
  echo EXIT SUCCESS
) | docker exec -i "%AGNO_ORACLE_CONTAINER%" sqlplus -s "sys/ai@localhost:1521/FREEPDB1 as sysdba"

if errorlevel 1 (
  echo Failed to configure the FREEPDB1 vector-memory allowance.
  exit /b 1
)

set "ORACLE_HOST=localhost"
set "ORACLE_PORT=%AGNO_ORACLE_PORT%"
set "ORACLE_SERVICE_NAME=FREEPDB1"
set "ORACLE_USER=ai"
set "ORACLE_PASSWORD=ai"
set "ORACLE_DB_URL=oracle+oracledb://ai:ai@localhost:%AGNO_ORACLE_PORT%/?service_name=FREEPDB1"

echo Database setup complete!
exit /b 0

:wait_for_oracle
for /l %%i in (1,1,120) do (
  docker exec "%AGNO_ORACLE_CONTAINER%" healthcheck.sh >nul 2>&1
  if not errorlevel 1 exit /b 0

  echo Waiting for Oracle: attempt %%i of 120
  timeout /t 5 /nobreak >nul
  if errorlevel 1 (
    echo Failed to wait between readiness checks.
    exit /b 1
  )
)

echo Oracle did not become ready after 120 attempts.
docker logs --tail 50 "%AGNO_ORACLE_CONTAINER%"
exit /b 1