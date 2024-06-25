USE ROLE engineer;
USE DATABASE dkurokawa;
USE SCHEMA trulens_demo;
USE WAREHOUSE dkurokawa;

DESCRIBE COMPUTE POOL dkurokawa_trulens_demo_compute_pool;

--DROP SERVICE dkurokawa_trulens_demo_app;
CREATE SERVICE dkurokawa_trulens_demo_app
    IN COMPUTE POOL dkurokawa_trulens_demo_compute_pool
    EXTERNAL_ACCESS_INTEGRATIONS = (dkurokawa_trulens_demo_access_integration)
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: dkurokawa-trulens-demo-app-container
        image: /dkurokawa/trulens_demo/dkurokawa_trulens_demo_image_repository/trulens_demo:latest
        env:
            SNOWFLAKE_ACCOUNT: "fab02971"
            SNOWFLAKE_DATABASE: "dkurokawa"
            SNOWFLAKE_SCHEMA: "trulens_demo"
            SNOWFLAKE_WAREHOUSE: "dkurokawa"
            SNOWFLAKE_ROLE: "engineer"
            SNOWFLAKE_CORTEX_SEARCH_SERVICE: "dkurokawa_trulens_demo_search_service"
            RUN_APP: "1"
        secrets:
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: username
            envVarName: SNOWFLAKE_USER
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: password
            envVarName: SNOWFLAKE_USER_PASSWORD
          - snowflakeSecret: dkurokawa.trulens_demo.replicate_api_token
            secretKeyRef: secret_string
            envVarName: REPLICATE_API_TOKEN
          - snowflakeSecret: dkurokawa.trulens_demo.pinecone_api_token
            secretKeyRef: secret_string
            envVarName: PINECONE_API_KEY
      endpoints:
      - name: trulens-demo-app-endpoint
        port: 8501
        public: true
    $$
;

--DROP SERVICE dkurokawa_trulens_demo_dashboard;
CREATE SERVICE dkurokawa_trulens_demo_dashboard
    IN COMPUTE POOL dkurokawa_trulens_demo_compute_pool
    EXTERNAL_ACCESS_INTEGRATIONS = (dkurokawa_trulens_demo_access_integration)
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: dkurokawa-trulens-demo-dashboard-container
        image: /dkurokawa/trulens_demo/dkurokawa_trulens_demo_image_repository/trulens_demo:latest
        env:
            SNOWFLAKE_ACCOUNT: "fab02971"
            SNOWFLAKE_DATABASE: "dkurokawa"
            SNOWFLAKE_SCHEMA: "trulens_demo"
            SNOWFLAKE_WAREHOUSE: "dkurokawa"
            SNOWFLAKE_ROLE: "engineer"
            SNOWFLAKE_CORTEX_SEARCH_SERVICE: "dkurokawa_trulens_demo_search_service"
            RUN_DASHBOARD: "1"
        secrets:
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: username
            envVarName: SNOWFLAKE_USER
          - snowflakeSecret: dkurokawa.trulens_demo.login_credentials
            secretKeyRef: password
            envVarName: SNOWFLAKE_USER_PASSWORD
          - snowflakeSecret: dkurokawa.trulens_demo.replicate_api_token
            secretKeyRef: secret_string
            envVarName: REPLICATE_API_TOKEN
          - snowflakeSecret: dkurokawa.trulens_demo.pinecone_api_token
            secretKeyRef: secret_string
            envVarName: PINECONE_API_KEY
      endpoints:
      - name: trulens-demo-dashboard-endpoint
        port: 8484
        public: true
    $$
;

SHOW SERVICES;
SELECT SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_app');
SELECT SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_dashboard');
DESCRIBE SERVICE dkurokawa_trulens_demo_app;
DESCRIBE SERVICE dkurokawa_trulens_demo_dashboard;
CALL SYSTEM$GET_SERVICE_LOGS('dkurokawa_trulens_demo_app', '0', 'dkurokawa-trulens-demo-app-container');
CALL SYSTEM$GET_SERVICE_LOGS('dkurokawa_trulens_demo_dashboard', '0', 'dkurokawa-trulens-demo-dashboard-container');
SHOW ENDPOINTS IN SERVICE dkurokawa_trulens_demo_app;
SHOW ENDPOINTS IN SERVICE dkurokawa_trulens_demo_dashboard;
CALL SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_app')
CALL SYSTEM$GET_SERVICE_STATUS('dkurokawa_trulens_demo_dashboard')
