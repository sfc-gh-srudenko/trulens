from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Sequence,
)

from snowflake.cortex import Complete
from trulens.feedback import llm_provider
from trulens.feedback import prompts as feedback_prompts
from trulens.providers.cortex import endpoint as cortex_endpoint

# If this is set, the provider will use this connection. This is useful for server-side evaluations which are done in a stored procedure and must have a single connection throughout the life of the stored procedure.
# TODO: This is a bit of a hack to pass the connection to the provider. Explore options on how to improve this.
_SNOWFLAKE_STORED_PROCEDURE_CONNECTION: Any = None
# Define the new version of the function


# def _patched_return_stream_response(
#     response: Response, deadline: Optional[float]
# ) -> dict:
#     client = SSEClient(response)
#     full_content = []  # Accumulate the content here
#     for event in client.events():
#         if deadline is not None and time.time() > deadline:
#             raise TimeoutError()
#         try:
#             message = json.loads(event.data)
#             full_content.append(message["choices"][0]["delta"]["content"])

#         except (json.JSONDecodeError, KeyError, IndexError):
#             # For the sake of evolution of the output format,
#             # ignore stream messages that don't match the expected format.
#             pass
#     final_message = {
#         "id": message["id"],
#         "created": message["created"],
#         "model": message["model"],
#         "tru_content": "".join(full_content),
#         "usage": message["usage"],
#     }
#     return final_message


# def _modified_complete_non_streaming_immediate(
#     model: str,
#     prompt,
#     options,
#     session=None,
#     deadline: Optional[float] = None,
# ) -> str:
#     response = _complete._complete_rest(
#         model=model,
#         prompt=prompt,
#         options=options,
#         session=session,
#         deadline=deadline,
#     )
#     return response


# # monkey patch the function to allow usage tracking from Cortex REST API
# _complete._return_stream_response = _patched_return_stream_response
# _complete._complete_non_streaming_immediate = (
#     _modified_complete_non_streaming_immediate
# )


class Cortex(
    llm_provider.LLMProvider
):  # require `pip install snowflake-ml-python` and a active Snowflake account with proper privileges
    # https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability

    DEFAULT_MODEL_ENGINE: ClassVar[str] = "llama3.1-8b"

    model_engine: str
    endpoint: cortex_endpoint.CortexEndpoint
    snowflake_conn: Any

    """Snowflake's Cortex COMPLETE endpoint. Defaults to `llama3.1-8b`.

    Reference: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex

    !!! example

        === "Connecting with user/password"
            ```python
            connection_parameters = {
                "account": <account>,
                "user": <user>,
                "password": <password>,
                "role": <role>,
                "database": <database>,
                "schema": <schema>,
                "warehouse": <warehouse>
            }
            provider = Cortex(snowflake.connector.connect(
                **connection_parameters
            ))
            ```

        === "Connecting with private key"
            ```python
            connection_parameters = {
                "account": <account>,
                "user": <user>,
                "private_key": <private_key>,
                "role": <role>,
                "database": <database>,
                "schema": <schema>,
                "warehouse": <warehouse>
            }
            provider = Cortex(snowflake.connector.connect(
                **connection_parameters
            ))
            ```

        === "Connecting with a private key file"
            ```python
            connection_parameters = {
                "account": <account>,
                "user": <user>,
                "private_key_file": <private_key_file>,
                "private_key_file_pwd": <private_key_file_pwd>,
                "role": <role>,
                "database": <database>,
                "schema": <schema>,
                "warehouse": <warehouse>
            }
            provider = Cortex(snowflake.connector.connect(
                **connection_parameters
            ))
            ```

    Args:
        snowflake_conn (Any): Snowflake connection. Note: This is not a snowflake session.

        model_engine (str, optional): Model engine to use. Defaults to `snowflake-arctic`.

    """

    def __init__(
        self,
        snowflake_conn: Any,
        model_engine: Optional[str] = None,
        *args,
        **kwargs: Dict,
    ):
        self_kwargs = dict(kwargs)

        self_kwargs["model_engine"] = (
            self.DEFAULT_MODEL_ENGINE if model_engine is None else model_engine
        )

        self_kwargs["endpoint"] = cortex_endpoint.CortexEndpoint(
            *args, **kwargs
        )

        # Create a Snowflake connector
        self_kwargs["snowflake_conn"] = _SNOWFLAKE_STORED_PROCEDURE_CONNECTION
        if _SNOWFLAKE_STORED_PROCEDURE_CONNECTION is None:
            self_kwargs["snowflake_conn"] = snowflake_conn
        if not callable(getattr(self_kwargs["snowflake_conn"], "cursor", None)):
            raise ValueError(
                "Invalid snowflake_conn: Expected a Snowflake connection object with a 'cursor' method. Please ensure you are not passing a session object."
            )
        super().__init__(**self_kwargs)

    def _invoke_cortex_complete(
        self,
        model: str,
        temperature: float,
        messages: Optional[Sequence[Dict]] = None,
    ) -> str:
        # Ensure messages are formatted as a JSON array string
        if messages is None:
            messages = []

        options = {"temperature": temperature}

        completion_res_str: str = Complete(
            model=model,
            prompt=messages,
            options=options,
        )
        return completion_res_str

        # completion_input_str = """
        #     SELECT SNOWFLAKE.CORTEX.COMPLETE(
        #         ?,
        #         parse_json(?),
        #         parse_json(?)
        #     )
        # """
        # if (
        #     hasattr(self.snowflake_conn, "_paramstyle")
        #     and self.snowflake_conn._paramstyle == "pyformat"
        # ):
        #     completion_input_str = completion_input_str.replace("?", "%s")

        # # Executing Snow SQL command requires an active snow session
        # cursor = self.snowflake_conn.cursor()
        # try:
        #     cursor.execute(
        #         completion_input_str,
        #         (model, messages_json_str, options_json_str),
        #     )
        #     result = cursor.fetchall()
        # finally:
        #     cursor.close()

        # return result

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

        completion_str = self._invoke_cortex_complete(**kwargs)

        return completion_str

    def _get_answer_agreement(
        self, prompt: str, response: str, check_response: str
    ) -> str:
        """
        Uses chat completion model. A function that completes a template to
        check if two answers agree.

        Args:
            text (str): A prompt to an agent.
            response (str): The agent's response to the prompt.
            check_response(str): The response to check against.

        Returns:
            str
        """

        assert self.endpoint is not None, "Endpoint is not set."

        messages = [
            {"role": "system", "content": feedback_prompts.AGREEMENT_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": check_response},
        ]

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            messages=messages,
        )
