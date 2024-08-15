import json
import os
from typing import ClassVar, Dict, Optional, Sequence

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
import snowflake
import snowflake.connector
from snowflake.connector import SnowflakeConnection
import streamlit as st
from trulens.feedback import LLMProvider
from trulens.providers.cortex.endpoint import CortexEndpoint

load_dotenv()


class Cortex(LLMProvider):
    # require `pip install snowflake-snowpark-python` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "snowflake-arctic"

    model_engine: str
    """Snowflake's Cortex COMPLETE endpoint. Defaults to `snowflake-arctic`.
       Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """

    endpoint: CortexEndpoint
    snowflake_conn: SnowflakeConnection

    def __init__(
        self, model_engine: Optional[str] = None, *args, **kwargs: Dict
    ):
        self_kwargs = dict(kwargs)

        self_kwargs["model_engine"] = (
            self.DEFAULT_MODEL_ENGINE if model_engine is None else model_engine
        )
        self_kwargs["endpoint"] = CortexEndpoint(*args, **kwargs)

        connection_parameters = {
            "account": st.secrets.get(
                "SNOWFLAKE_ACCOUNT", os.environ.get("SNOWFLAKE_ACCOUNT")
            ),
            "user": st.secrets.get(
                "SNOWFLAKE_USER", os.environ.get("SNOWFLAKE_USER")
            ),
            "role": st.secrets.get(
                "SNOWFLAKE_ROLE", os.environ.get("SNOWFLAKE_ROLE")
            ),
            "database": st.secrets.get(
                "SNOWFLAKE_DATABASE", os.environ.get("SNOWFLAKE_DATABASE")
            ),
            "schema": st.secrets.get(
                "SNOWFLAKE_SCHEMA", os.environ.get("SNOWFLAKE_SCHEMA")
            ),
            "warehouse": st.secrets.get(
                "SNOWFLAKE_WAREHOUSE", os.environ.get("SNOWFLAKE_WAREHOUSE")
            ),
        }

        if (
            "SNOWFLAKE_USER_PASSWORD" in st.secrets
            or "SNOWFLAKE_USER_PASSWORD" in os.environ
        ):
            connection_parameters["password"] = st.secrets.get(
                "SNOWFLAKE_USER_PASSWORD",
                os.environ.get("SNOWFLAKE_USER_PASSWORD"),
            )

        if (
            "SNOWFLAKE_PRIVATE_KEY" in st.secrets
            or "SNOWFLAKE_PRIVATE_KEY" in os.environ
        ):
            # Retrieve the private key as a string from st.secrets or environment
            private_key_str = st.secrets.get(
                "SNOWFLAKE_PRIVATE_KEY", os.getenv("SNOWFLAKE_PRIVATE_KEY")
            )

            # Convert the string to bytes
            private_key_bytes = private_key_str.encode()

            connection_parameters["private_key"] = (
                serialization.load_pem_private_key(
                    private_key_bytes, password=None, backend=default_backend()
                )
            )

        if (
            "SNOWFLAKE_PRIVATE_KEY_FILE" in st.secrets
            or "SNOWFLAKE_PRIVATE_KEY_FILE" in os.environ
        ):
            connection_parameters["private_key_file"] = st.secrets.get(
                "SNOWFLAKE_PRIVATE_KEY_FILE",
                os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
            )

        if (
            "SNOWFLAKE_PRIVATE_KEY_FILE_PWD" in st.secrets
            or "SNOWFLAKE_PRIVATE_KEY_FILE_PWD" in os.environ
        ):
            connection_parameters["private_key_file_pwd"] = st.secrets.get(
                "SNOWFLAKE_PRIVATE_KEY_FILE_PWD",
                os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"),
            )

        # Create a Snowflake connector
        self_kwargs["snowflake_conn"] = snowflake.connector.connect(
            **connection_parameters
        )
        super().__init__(**self_kwargs)

    def _exec_snowsql_complete_command(
        self,
        model: str,
        temperature: float,
        messages: Optional[Sequence[Dict]] = None,
    ):
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []
        messages_json_str = json.dumps(messages)

        options = {"temperature": temperature}
        options_json_str = json.dumps(options)

        completion_input_str = """
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                %s,
                parse_json(%s),
                parse_json(%s)
            )
        """

        # Executing Snow SQL command requires an active snow session
        cursor = self.snowflake_conn.cursor()
        try:
            cursor.execute(
                completion_input_str,
                (model, messages_json_str, options_json_str),
            )
            result = cursor.fetchall()
        finally:
            cursor.close()

        return result

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs,
    ) -> str:
        if "model" not in kwargs:
            kwargs["model"] = self.model_engine
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        if messages is not None:
            kwargs["messages"] = messages

        elif prompt is not None:
            kwargs["messages"] = [{"role": "system", "content": prompt}]
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        res = self._exec_snowsql_complete_command(**kwargs)

        completion = json.loads(res[0][0])["choices"][0]["messages"]

        return completion
