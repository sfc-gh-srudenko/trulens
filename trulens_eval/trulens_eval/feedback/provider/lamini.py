import logging
import os
from pprint import pformat
from typing import Any, ClassVar, Dict, Optional, Sequence

import pydantic

from trulens_eval.feedback.provider.base import LLMProvider
from trulens_eval.feedback.provider.base import OutputType
from trulens_eval.feedback.provider.base import WithOutputType
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LAMINI

with OptionalImports(messages=REQUIREMENT_LAMINI):
    import lamini

    from trulens_eval.feedback.provider.endpoint.lamini import LaminiEndpoint

# check that the optional imports are not dummies:
OptionalImports(messages=REQUIREMENT_LAMINI).assert_installed(lamini)

logger = logging.getLogger(__name__)


class Lamini(WithOutputType, LLMProvider):
    """Out of the box feedback functions calling Lamini API.

    Create an Lamini Provider with out of the box feedback functions. Lamini
    supports output type specification making it more efficient at some
    tasks/feedback functions.

    Usage:
        ```python
        from trulens_eval.feedback.provider.lamini import Lamini
        lamini_provider = Lamini()
        ```
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "mistralai/Mistral-7B-Instruct-v0.1"

    model_engine: str = pydantic.Field(alias="model_name")
    """Model specification of parent class.
    
    We alias to `model_name` to match lamini terminology. Defaults to
    `mistralai/Mistral-7B-Instruct-v0.1`.
    """

    model_name: str = DEFAULT_MODEL_NAME
    """The Lamini completion model. Defaults to `mistralai/Mistral-7B-Instruct-v0.1`.
    
    List can be found in (lamini model docs
    page)[https://lamini-ai.github.io/inference/models_list/].
    
    """

    generation_args: Dict[str, Any] = pydantic.Field(default_factory=dict)
    """Additional arguments to pass to the `Lamini.generate` as needed for
    model/usage.

    !!! Warning:
        Feedback functions override the `output_type` argument to
        `Lamini.generate` so this parameter cannot be set using
        `generation_args`.
    """

    endpoint: Endpoint

    def __init__(
        self,
        model_name: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        endpoint: Optional[Endpoint] = None,
        **kwargs: dict
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        if model_name is None:
            model_name = self.DEFAULT_MODEL_NAME

        if generation_kwargs is None:
            generation_kwargs = {}

        if 'output_type' in generation_kwargs:
            raise ValueError(
                "`output_type` cannot be set for `generation_args` as it is overwritten by each feedback function."
            )

        self_kwargs = {}
        self_kwargs.update(**kwargs)
        self_kwargs['model_name'] = model_name
        self_kwargs['generation_args'] = generation_kwargs
        self_kwargs['endpoint'] = LaminiEndpoint(
            **kwargs
        )

        if lamini.api_key is None: #  and os.environ.get("LAMINI_API_KEY") is None:
            logger.warning("No lamini API key is set. You may need to set lamini.api_key before using this provider.")

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        output_type: Optional[OutputType] = "string",
        **kwargs
    ) -> str:

        if lamini.api_key is None:
            raise ValueError("Lamini API key is not set. Please set lamini.api_key before using the lamini provider.")

        lamini_instance = lamini.Lamini(model_name=self.model_name)

        if output_type is None:
            output_type = "string"

        if prompt is not None:
            pass
        elif messages is not None:
            # Assume there is only one system message.
            if len(messages) > 1:
                raise ValueError(
                    "Lamini only supports a single system message in a single completion."
                )
            prompt=messages[0]['content']
        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        all_args = dict(
                output_type={'output': output_type},
                **kwargs,
                **self.generation_args
            )

        comp = lamini_instance.generate(prompt, **all_args)

        if "output" not in comp:
            raise ValueError(f"Unexpected response from lamini: {comp}")

        comp = comp['output']

        if output_type != "string":
            comp = str(comp)

        return comp