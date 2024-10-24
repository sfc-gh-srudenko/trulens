"""
# Dev Notes

This class makes use of langchain's cost tracking for openai models. Changes to
the involved classes will need to be adapted here. The important classes are:

- `langchain.schema.LLMResult`
- `langchain.callbacks.openai_info.OpenAICallbackHandler`

## Changes for openai 1.0

- Previously we instrumented classes `openai.*` and their methods `create` and
  `acreate`. Now we instrument classes `openai.resources.*` and their `create`
  methods. We also instrument `openai.resources.chat.*` and their `create`. To
  be determined is the instrumentation of the other classes/modules under
  `openai.resources`.

- openai methods produce structured data instead of dicts now. langchain expects
  dicts so we convert them to dicts.

"""

import inspect
import logging
import pprint
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from langchain.schema import Generation
from langchain.schema import LLMResult
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
import pydantic
from pydantic.v1 import BaseModel as v1BaseModel
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.schema import base as base_schema
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import pace as pace_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing import feature as otel_tracing_feature

import openai
from openai import resources
from openai.resources import chat
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()

TOpenAIReturn = Union[
    openai.types.completion.Completion,
    openai.Stream[openai.types.completion.Completion],
    openai.types.chat.chat_completion.ChatCompletion,
    openai.Stream[openai.types.chat.chat_completion_chunk.ChatCompletionChunk],
    openai.types.create_embedding_response.CreateEmbeddingResponse,
    openai.types.moderation.Moderation,
]
"""Types that openai respones can attain, or at least the ones we handle in cost tracking."""

TOpenAIResponse = TOpenAIReturn

T = TypeVar("T")  # TODO bound


class OpenAIClient(serial_utils.SerialModel):
    """A wrapper for openai clients.

    This class allows wrapped clients to be serialized into json. Does not
    serialize API key though. You can access openai.OpenAI under the `client`
    attribute. Any attributes not defined by this wrapper are looked up from the
    wrapped `client` so you should be able to use this instance as if it were an
    `openai.OpenAI` instance.
    """

    REDACTED_KEYS: ClassVar[List[str]] = ["api_key", "default_headers"]
    """Parameters of the OpenAI client that will not be serialized because they
    contain secrets."""

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    client: Union[openai.OpenAI, openai.AzureOpenAI] = pydantic.Field(
        exclude=True
    )
    """Deserialized representation."""

    client_cls: pyschema_utils.Class
    """Serialized representation class."""

    client_kwargs: dict
    """Serialized representation constructor arguments."""

    def __init__(
        self,
        client: Optional[Union[openai.OpenAI, openai.AzureOpenAI]] = None,
        client_cls: Optional[pyschema_utils.Class] = None,
        client_kwargs: Optional[dict] = None,
    ):
        if client_kwargs is not None:
            # Check if any of the keys which will be redacted when serializing
            # were set and give the user a warning about it.
            for rkey in OpenAIClient.REDACTED_KEYS:
                if rkey in client_kwargs:
                    logger.warning(
                        "OpenAI parameter %s is not serialized for DEFERRED feedback mode. "
                        "If you are not using DEFERRED, you do not need to do anything. "
                        "If you are using DEFERRED, try to specify this parameter through env variable or another mechanism.",
                        rkey,
                    )

        if client is None:
            if client_kwargs is None and client_cls is None:
                client = openai.OpenAI()

            elif client_kwargs is None or client_cls is None:
                raise ValueError(
                    "`client_kwargs` and `client_cls` are both needed to deserialize an openai.`OpenAI` client."
                )

            else:
                if isinstance(client_cls, dict):
                    # TODO: figure out proper pydantic way of doing these things. I
                    # don't think we should be required to parse args like this.
                    client_cls = pyschema_utils.Class.model_validate(client_cls)

                cls = client_cls.load()

                timeout = client_kwargs.get("timeout")
                if timeout is not None:
                    client_kwargs["timeout"] = openai.Timeout(**timeout)

                client = cls(**client_kwargs)

        if client_cls is None:
            assert client is not None

            client_class = type(client)

            # Recreate constructor arguments and store in this dict.
            client_kwargs = {}

            # Guess the constructor arguments based on signature of __new__.
            sig = inspect.signature(client_class.__init__)

            for k, _ in sig.parameters.items():
                if k in OpenAIClient.REDACTED_KEYS:
                    # Skip anything that might have the api_key in it.
                    # default_headers contains the api_key.
                    continue

                if python_utils.safe_hasattr(client, k):
                    client_kwargs[k] = pyschema_utils.safe_getattr(client, k)

            # Create serializable class description.
            client_cls = pyschema_utils.Class.of_class(client_class)

        super().__init__(
            client=client, client_cls=client_cls, client_kwargs=client_kwargs
        )

    def __getattr__(self, k):
        # Pass through attribute lookups to `self.client`, the openai.OpenAI
        # instance.
        if python_utils.safe_hasattr(self.client, k):
            return pyschema_utils.safe_getattr(self.client, k)

        raise AttributeError(
            f"No attribute {k} in wrapper OpenAiClient nor the wrapped OpenAI client."
        )


class OpenAICallback(core_endpoint.EndpointCallback):
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    langchain_handler: OpenAICallbackHandler = pydantic.Field(
        default_factory=OpenAICallbackHandler, exclude=True
    )

    chunks: List[Generation] = pydantic.Field(
        default_factory=list,
        exclude=True,
    )

    _FIELDS_MAP: ClassVar[List[Tuple[str, str]]] = [
        ("cost", "total_cost"),
        ("n_tokens", "total_tokens"),
        ("n_successful_requests", "successful_requests"),
        ("n_prompt_tokens", "prompt_tokens"),
        ("n_completion_tokens", "completion_tokens"),
    ]
    """Pairs where first element is the cost attribute name and second is
    attribute of langchain.OpenAICallbackHandler that corresponds to it."""

    def handle_generation_chunk(self, response: Any) -> None:
        super().handle_generation_chunk(response=response)

        try:
            if hasattr(response, "choices"):
                choices = response.choices

                for choice in choices:
                    if choice.finish_reason == "stop":
                        llm_result = LLMResult(
                            llm_output=dict(
                                token_usage={}, model_name=response.model
                            ),
                            generations=[self.chunks],
                        )
                        self.chunks = []
                        self.handle_generation(response=llm_result)
                    else:
                        if hasattr(choice, "delta"):
                            self.chunks.append({"text": choice.delta.content})

        finally:
            return response

    def handle_generation(self, response: LLMResult) -> None:
        super().handle_generation(response)

        self.langchain_handler.on_llm_end(response)

        addl_cost = base_schema.Cost(**{
            cost_field: getattr(self.langchain_handler, langchain_field)
            for (
                cost_field,
                langchain_field,
            ) in OpenAICallback._FIELDS_MAP
        })

        # n_successful_requests comes from langchain handler.

        self.cost += addl_cost

    def handle_embedding(self, response: Any) -> None:
        super().handle_embedding(response)

        self.cost.n_successful_requests += 1
        self.cost.n_embeddings += len(response.data)
        # TODO: there seems to be usage info in these responses sometimes as well


if otel_tracing_feature._FeatureSetup.are_optionals_installed():
    from trulens.experimental.otel_tracing.core.feedback import (
        endpoint as experimental_core_endpoint,
    )

    class WrapperOpenAICallback(
        experimental_core_endpoint._WrapperEndpointCallback[
            TOpenAIReturn, TOpenAIResponse
        ]
    ):
        """EXPERIMENTAL(otel_tracing): process the results of instrumented
        openai calls to extract cost information."""

        # _WrapperCallableCallback optional
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # increments n_requests

            self.langchain_handler: OpenAICallbackHandler = (
                OpenAICallbackHandler()
            )

            self.chunks: List[Any] = []

            name = self.func.__qualname__

            if name.startswith("Completions."):
                self.cost.n_completion_requests += 1
            elif name.startswith("Embeddings."):
                self.cost.n_embedding_requests += 1
            elif name.startswith("Moderations."):
                self.cost.n_classification_requests += 1
            else:
                logger.warning(
                    "Unknown openai call %s. Cost tracking will not be available.",
                    self.func.__qualname__,
                )

        # _WrapperCallableCallback optional
        def on_callable_return(
            self,
            ret: TOpenAIReturn,
        ) -> TOpenAIReturn:
            """Process a returned response from an openai call.

            As there are multiple types of calls being handled here, we need to make
            various checks to see what sort of data to process based on the call
            made.
            """

            name = self.func.__qualname__

            if name.startswith("Completions."):
                if isinstance(ret, openai.Stream):
                    # TODO: wrap this in a generator that will call on_endpoint_generation_chunk.
                    pass

                elif isinstance(
                    ret,
                    (
                        openai.types.completion.Completion,
                        openai.types.chat.chat_completion.ChatCompletion,
                    ),
                ):
                    # Don't bother processing these and instead wait for chunks to be processed.
                    self.on_endpoint_generation(response=ret)
                else:
                    logger.warning(
                        "Unknown openai completion response type %s.",
                        type(ret),
                    )

            elif name.startswith("Embeddings."):
                if isinstance(
                    ret,
                    openai.types.create_embedding_response.CreateEmbeddingResponse,
                ):
                    self.on_endpoint_embedding(response=ret)
                else:
                    logger.warning(
                        "Unknown openai embedding response type %s.",
                        type(ret),
                    )

            elif name.startswith("Moderations."):
                if isinstance(ret, openai.types.moderation.Moderation):
                    self.on_endpoint_classification(response=ret)
                else:
                    logger.warning(
                        "Unknown openai moderation response type %s.",
                        type(ret),
                    )

            else:
                logger.warning(
                    "Unknown openai call %s. Cost tracking will not be available.",
                    self.func.__qualname__,
                )

            return ret

        def on_endpoint_embedding(
            self,
            response: openai.types.create_embedding_response.CreateEmbeddingResponse,
        ) -> None:
            super().on_endpoint_embedding(response)

            self.cost.n_embeddings += len(response.data)

        def on_endpoint_classification(
            self, response: openai.types.moderation.Moderation
        ) -> None:
            """Extract costs from a moderation classification response."""

            super().on_endpoint_classification(response)

            self.cost.n_classes += len(response.categories.model_fields)

        # TODO: Finish integrating the below into the other methods here.
        """
        # _WrapperEndpointCallback optional
        def on_endpoint_generation(
            self,
            response: Union[
                openai.types.completion.Completion,
                openai.types.chat.chat_completion.ChatCompletion,
            ],
        ) -> None:
            super().on_endpoint_generation(response)

            bindings = self.bindings

            model_name = ""
            if "model" in bindings.kwargs:
                model_name = bindings.kwargs["model"]

            if isinstance(response, openai.Stream):
                # NOTE(piotrm): Merely checking membership in these will exhaust internal
                # genertors or iterators which will break users' code. While we work
                # out something, I'm disabling any cost-tracking for these streams.
                logger.warning("Cannot track costs from a OpenAI Stream.")
                return

            counted_something = False
            if hasattr(response, "usage"):
                counted_something = True

                if isinstance(response.usage, pydantic.BaseModel):
                    usage = response.usage.model_dump()
                elif isinstance(ret.usage, pydantic.v1.BaseModel):
                    usage = ret.usage.dict()
                elif isinstance(ret.usage, Dict):
                    usage = ret.usage
                else:
                    usage = None

                # See how to construct in langchain.llms.openai.OpenAIChat._generate
                llm_res = LLMResult(
                    generations=[[]],
                    llm_output=dict(token_usage=usage, model_name=model_name),
                    run=None,
                )

                self.on_endpoint_generation(response=llm_res)

            if "choices" in ret and "delta" in ret.choices[0]:
                # Streaming data.
                content = ret.choices[0].delta.content

                gen = Generation(text=content or "", generation_info=ret)
                self.on_endpoint_generation_chunk(gen)

                counted_something = True

            if not counted_something:
                logger.warning(
                    "Could not find usage information in openai response:\n%s",
                    pp.pformat(ret),
                )

            return
        """

        # WrapperEndpointCallback optional
        def on_endpoint_generation_chunk(
            self, response: TOpenAIResponse
        ) -> None:
            """Process a generation chunk."""

            super().on_endpoint_generation_chunk(response=response)

            self.chunks.append(response)

            if response.choices[0].finish_reason == "stop":
                llm_result = LLMResult(
                    llm_output=dict(
                        token_usage=dict(), model_name=response.model
                    ),
                    generations=[self.chunks],
                )
                self.chunks = []
                self.on_endpoint_generation(response=llm_result)

        # WrapperEndpointCallback optional
        def on_endpoint_generation(self, response: TOpenAIResponse) -> None:
            """Process a generation/completion."""

            super().on_endpoint_generation(response)

            self.langchain_handler.on_llm_end(response)

            assert self.cost is not None

            self.cost += base_schema.Cost(**{
                cost_field: getattr(self.langchain_handler, langchain_field, 0)
                for cost_field, langchain_field in [
                    ("cost", "total_cost"),
                    ("n_tokens", "total_tokens"),
                    ("n_successful_requests", "successful_requests"),
                    ("n_prompt_tokens", "prompt_tokens"),
                    ("n_completion_tokens", "completion_tokens"),
                ]
            })


class OpenAIEndpoint(core_endpoint.Endpoint):
    """OpenAI endpoint.

    Instruments "create" methods in openai client.

    Args:
        client: openai client to use. If not provided, a new client will be
            created using the provided kwargs.

        **kwargs: arguments to constructor of a new OpenAI client if `client`
            not provided.

    """

    client: OpenAIClient

    def __init__(
        self,
        client: Optional[
            Union[openai.OpenAI, openai.AzureOpenAI, OpenAIClient]
        ] = None,
        rpm: Optional[int] = None,
        pace: Optional[pace_utils.Pace] = None,
        **kwargs: dict,
    ):
        self_kwargs = {
            "rpm": rpm,
            "pace": pace,
            **kwargs,
        }

        self_kwargs["callback_class"] = OpenAICallback

        if constant_utils.CLASS_INFO in kwargs:
            del kwargs[constant_utils.CLASS_INFO]

        if client is None:
            # Pass kwargs to client.
            client = openai.OpenAI(**kwargs)
            self_kwargs["client"] = OpenAIClient(client=client)

        else:
            if len(kwargs) != 0:
                logger.warning(
                    "Arguments %s are ignored as `client` was provided.",
                    list(kwargs.keys()),
                )

            # Convert openai client to our wrapper if needed.
            if not isinstance(client, OpenAIClient):
                assert isinstance(
                    client, (openai.OpenAI, openai.AzureOpenAI)
                ), "OpenAI client expected"

                client = OpenAIClient(client=client)

            self_kwargs["client"] = client

        # for pydantic.BaseModel
        super().__init__(**self_kwargs)

        self._instrument_module_members(openai, "create")
        self._instrument_module_members(resources, "create")
        self._instrument_module_members(chat, "create")

    def handle_wrapped_call(
        self,
        func: Callable,
        bindings: inspect.BoundArguments,
        response: Any,
        callback: Optional[core_endpoint.EndpointCallback],
    ) -> Any:
        # TODELETE(otel_tracing). Delete once otel_tracing is no longer
        # experimental.

        # TODO: cleanup/refactor. This method inspects the results of an
        # instrumented call made by an openai client. As there are multiple
        # types of calls being handled here, we need to make various checks to
        # see what sort of data to process based on the call made.

        # Generic lazy value should have already been taken care of by base Endpoint class.
        assert not python_utils.is_lazy(response)

        context_vars = {
            core_endpoint.Endpoint._context_endpoints: core_endpoint.Endpoint._context_endpoints.get()
        }

        if isinstance(response, (openai.AsyncStream, openai.Stream)):
            # Don't bother processing these and instead wait for chunks to be processed.

            def handle_chunk(chunk: T) -> T:
                with python_utils.with_context(context_vars):
                    self.global_callback.handle_generation_chunk(chunk)

                    if callback is not None:
                        callback.handle_generation_chunk(chunk)

                    return chunk

            if isinstance(response, openai.AsyncStream):
                response._iterator = python_utils.wrap_async_generator(
                    response._iterator,
                    wrap=handle_chunk,
                    context_vars=context_vars,
                )
                return response

            if isinstance(response, openai.Stream):
                response._iterator = python_utils.wrap_generator(
                    response._iterator,
                    wrap=handle_chunk,
                    context_vars=context_vars,
                )
                return response

        model_name = ""
        if "model" in bindings.kwargs:
            model_name = bindings.kwargs["model"]

        results = None
        if "results" in response:
            results = response["results"]

        counted_something = False
        if hasattr(response, "usage"):
            counted_something = True

            if isinstance(response.usage, pydantic.BaseModel):
                usage = response.usage.model_dump()
            elif isinstance(response.usage, v1BaseModel):
                usage = response.usage.dict()
            elif isinstance(response.usage, Dict):
                usage = response.usage
            else:
                usage = None

            if isinstance(response, ChatCompletion):
                # See how to construct in langchain.llms.openai.OpenAIChat._generate
                llm_res = LLMResult(
                    generations=[[]],
                    llm_output=dict(token_usage=usage, model_name=model_name),
                    run=None,
                )

                self.global_callback.handle_generation(response=llm_res)

                if callback is not None:
                    callback.handle_generation(response=llm_res)

            elif isinstance(response, CreateEmbeddingResponse):
                self.global_callback.handle_embedding(response=response)

                if callback is not None:
                    callback.handle_embedding(response=response)

            else:
                logger.warning(
                    "Unknown openai response type with usage information:\n%s",
                    pp.pformat(response),
                )

        if "choices" in response:
            if "delta" in response.choices[0]:
                # Streaming data.
                content = response.choices[0].delta.content

                gen = Generation(text=content or "", generation_info=response)
                self.global_callback.handle_generation_chunk(gen)
                if callback is not None:
                    callback.handle_generation_chunk(gen)

                counted_something = True

            else:
                pass
                # Async responses that are not streams are like this. Should
                # already be handled by the "usage" above.

        if results is not None:
            for res in results:
                if "categories" in res:
                    counted_something = True
                    self.global_callback.handle_classification(response=res)

                    if callback is not None:
                        callback.handle_classification(response=res)

        if not counted_something:
            logger.warning(
                "Could not find usage information in openai response:\n%s",
                pp.pformat(response),
            )

        return response
