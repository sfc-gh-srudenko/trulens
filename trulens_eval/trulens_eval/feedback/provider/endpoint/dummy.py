from __future__ import annotations

import inspect
import logging
from pprint import PrettyPrinter
import random
import sys
from time import sleep
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

from numpy import random as np_random
from pydantic import Field

from trulens_eval.feedback.provider.endpoint.base import DEFAULT_RPM
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.feedback.provider.endpoint.base import EndpointCallback
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.threading import DEFAULT_NETWORK_TIMEOUT

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")

class DummyAPI(object):
    """A dummy model evaluation API used by DummyEndpoint.
    
    This is also used for testing cost tracking instrumentation.
    """

    loading_prob: float
    """How often to produce the "model loading" response that huggingface api
    sometimes produces."""

    loading_time: Callable[[], float] = \
        Field(exclude=True, default_factory=lambda: lambda: random.uniform(0.73, 3.7))
    """How much time to indicate as needed to load the model."""

    error_prob: float
    """How often to produce an error response."""

    freeze_prob: float
    """How often to freeze instead of producing a response."""

    overloaded_prob: float
    """# How often to produce the overloaded message that huggingface sometimes produces."""

    alloc: int
    """How much data in bytes to allocate when making requests."""

    delay: float = 0.0
    """How long to delay each request."""

    def __init__(
        self,
        error_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        alloc: int = 1024 * 1024,
        delay: float = 0.0,
        **kwargs
    ):
        assert error_prob + freeze_prob + overloaded_prob + loading_prob <= 1.0, "Probabilites should not exceed 1.0 ."
        assert alloc >= 0
        assert delay >= 0.0

    def post(self, url: str, payload: JSON, timeout: Optional[float] = None) -> Dict:
        """Pretend to make an http post request to some model execution API."""

        if timeout is None:
            timeout = DEFAULT_NETWORK_TIMEOUT

        # allocate some data to pretend we are doing hard work
        temporary = [0x42] * self.alloc
        # Use `temporary`` to make sure it doesn't get compiled away.
        logger.debug("I have allocated %s bytes.", sys.getsizeof(temporary))

        if self.delay > 0.0:
            sleep(max(0.0, np_random.normal(self.delay, self.delay / 2)))

        r = random.random()
        j: JSON = {}

        if r < self.freeze_prob:
            # Simulated freeze outcome.

            while True:
                sleep(timeout)
                raise TimeoutError()

        r -= self.freeze_prob

        if r < self.error_prob:
            # Simulated error outcome.

            raise RuntimeError("Simulated error happened.")
        r -= self.error_prob

        if r < self.loading_prob:
            # Simulated loading model outcome.

            j = {'estimated_time': self.loading_time()}
        r -= self.loading_prob

        if r < self.overloaded_prob:
            # Simulated overloaded outcome.

            j = {'error': "overloaded"}
        r -= self.overloaded_prob

         # The rest is the same as in Endpoint:

        # Huggingface public api sometimes tells us that a model is loading and
        # how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            logger.warning(
                "Waiting for %s (%s) second(s).",
                j,
                wait_time,
            )
            sleep(wait_time + 2)
            return self.post(url, payload, timeout=timeout)

        if isinstance(j, Dict) and "error" in j:
            error = j['error']
            if error == "overloaded":
                logger.warning(
                    "Waiting for overloaded API before trying again."
                )
                sleep(10)
                return self.post(url, payload, timeout=timeout)

            raise RuntimeError(error)

        assert isinstance(
            j, Sequence
        ) and len(j) > 0, f"Post did not return a sequence: {j}"

        j['status'] = 'success'
        return j

    def completion(self, *args, model: str, text: str, **kwargs) -> Dict:
        """Fake text completion request."""

        # Fake http post request, might raise an exception or cause delays.
        postret = self.post(
            url="https://fakeservice.com/classify",
            payload={'mode': 'completion', 'model': model, 'text': text}
        )

        if postret.get('status', 'failure') != 'success':
            raise RuntimeError("Failed to generate text completion.")

        generated_text: str = "my original response is " + text

        result = {
            'generated_text': generated_text,
            'usage': {
                # Fake usage information.
                'n_tokens': len(generated_text.split()) + len(text.split()),
                'n_prompt_tokens': len(text.split()),
                'n_completion_tokens': len(generated_text.split()),
                'cost': len(generated_text) * 0.0002 + len(text.split()) * 0.0001
            }
        }

        return result

    def classify(self, *args, model: str = "fakeclassier", text: str, **kwargs) -> Dict:
        """Fake classification request."""

        # Fake http post request, might raise an exception or cause delays.
        postret = self.post(
            url="https://fakeservice.com/classify",
            payload={'mode': 'classification', 'model': model, 'text': text}
        )

        if postret.get('status', 'failure') != 'success':
            raise RuntimeError("Failed to generate text completion.")

        # Simulated success outcome with some constant results plus some randomness.
        result = {
            'status': 'success',
            'scores': [
            {
                'label': 'LABEL_1',
                'score': 0.6034979224205017 + random.random()
            }, {
                'label': 'LABEL_2',
                'score': 0.2648237645626068 + random.random()
            }, {
                'label': 'LABEL_0',
                'score': 0.13167837262153625 + random.random()
            }
        ]}

        return result

class DummyEndpointCallback(EndpointCallback):

    def on_call(self, func: Callable, args: Tuple[str], kwargs: Dict[str, Any]):
        self.cost.n_requests += 1

    def on_return(self, func: Callable[..., Any], bindings: inspect.BoundArguments, ret: Any):
        self.cost.n_successful_requests += 1

        if "usage" in ret:
            # fake completion
            usage = ret["usage"]
            self.cost.cost += usage.get("cost", 0.0)
            self.cost.n_tokens += usage.get("n_tokens", 0)
            self.cost.n_prompt_tokens += usage.get("n_prompt_tokens", 0)
            self.cost.n_completion_tokens += usage.get("n_completion_tokens", 0)
        else:
            # fake classification
            self.cost.n_classes += len(ret.get("scores", []))
        
    def on_iteration(self):
        self.cost.n_stream_chunks += 1

class DummyEndpoint(Endpoint):
    """Endpoint for testing purposes.
    
    Does not make any network calls and just pretends to.
    """

    def __new__(cls, *args, **kwargs):
        return super(Endpoint, cls).__new__(cls, name="dummyendpoint")

    def __init__(
        self,
        name: str = "dummyendpoint",
        rpm: float = DEFAULT_RPM * 10,
        **kwargs
    ):
        if safe_hasattr(self, "callback_class"):
            # Already created with SingletonPerName mechanism
            return

        assert rpm > 0

        kwargs['name'] = name
        kwargs['callback_class'] = DummyEndpointCallback

        super().__init__(
            **kwargs, **locals_except("self", "name", "kwargs", "__class__")
        )

        logger.info(
            "Using DummyEndpoint with %s",
            locals_except('self', 'name', 'kwargs', '__class__')
        )


DummyEndpoint.model_rebuild()