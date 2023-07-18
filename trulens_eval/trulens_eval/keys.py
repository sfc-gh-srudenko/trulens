"""
# API keys and configuration 

## Setting keys

To check whether appropriate api keys have been set:

```python 
from trulens_eval.keys import check_keys

check_keys(
    "OPENAI_API_KEY",
    "HUGGINGFACE_API_KEY"
)
```

Alternatively you can set using `check_or_set_keys`:

```python 
from trulens_eval.keys import check_or_set_keys

check_or_set_keys(
    OPENAI_API_KEY="to fill in", 
    HUGGINGFACE_API_KEY="to fill in"
)
```

This line checks that you have the requisite api keys set before continuing the
notebook. They do not need to be provided, however, right on this line. There
are several ways to make sure this check passes:

- *Explicit* -- Explicitly provide key values to `check_keys`.

- *Python* -- Define variables before this check like this:

```python
OPENAI_API_KEY="something"
```

- *Environment* -- Set them in your environment variable. They should be visible when you execute:

```python
import os
print(os.environ)
```

- *.env* -- Set them in a .env file in the same folder as the example notebook or one of
  its parent folders. An example of a .env file is found in
  `trulens_eval/trulens_eval/env.example` .

- *3rd party* -- For some keys, set them as arguments to the 3rd-party endpoint class. For
  example, with `openai`, do this ahead of the `check_keys` check:

```python
import openai
openai.api_key = "something"
```

- *Endpoint class* For some keys, set them as arguments to trulens_eval endpoint class that
  manages the endpoint. For example, with `openai`, do this ahead of the
  `check_keys` check:

```python
from trulens_eval.provider_apis import OpenAIEndpoint
openai_endpoint = OpenAIEndpoint(api_key="something")
```

- *Provider class* For some keys, set them as arguments to trulens_eval feedback
  collection ("provider") class that makes use of the relevant endpoint. For
  example, with `openai`, do this ahead of the `check_keys` check:

```python
from trulens_eval.feedback import OpenAI
openai_feedbacks = OpenAI(api_key="something")
```

In the last two cases, please note that the settings are global. Even if you
create multiple OpenAI or OpenAIEndpoint objects, they will share the
configuration of keys (and other openai attributes).

## Other API attributes

Some providers may require additional configuration attributes beyond api key.
For example, `openai` usage via azure require special keys. To set those, you
should use the 3rd party class method of configuration. For example with
`openai`:

```python
import openai

openai.api_type = "azure"
openai.api_key = "..."
openai.api_base = "https://example-endpoint.openai.azure.com"
openai.api_version = "2023-05-15"  # subject to change
# See https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/switching-endpoints .
```

Our example notebooks will only check that the api_key is set but will make use
of the configured openai object as needed to compute feedback.
"""

from collections import defaultdict
import logging
import os
from pathlib import Path, PurePath
from typing import Tuple

import cohere
import dotenv

from trulens_eval.util import caller_frame
from trulens_eval.util import UNICODE_CHECK
from trulens_eval.util import UNICODE_STOP

logger = logging.getLogger(__name__)


def get_config_file() -> Path:
    """
    Looks for a .env file in current folder or its parents. Returns Path of
    found .env or None if not found.
    """
    for path in [Path().cwd(), *Path.cwd().parents]:
        file = path / ".env"
        if file.exists():
            return file

    return None

def get_config() -> Tuple[Path, dict]:
    config_file = get_config_file()
    if config_file is None:
        logger.warning(
            f"No .env found in {Path.cwd()} or its parents. "
            "You may need to specify secret keys in another manner."
        )
        return None, None
    else:
        return config_file, dotenv.dotenv_values(config_file)

def set_openai_key() -> None:
    """
    Sets the openai class attribute `api_key` to its value from the
    OPENAI_API_KEY env var.
    """

    if 'OPENAI_API_KEY' in os.environ:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]


global cohere_agent
cohere_agent = None


def get_cohere_agent() -> cohere.Client:
    """
    Gete a singleton cohere agent. Sets its api key from env var COHERE_API_KEY.
    """

    global cohere_agent
    if cohere_agent is None:
        cohere.api_key = os.environ['COHERE_API_KEY']
        cohere_agent = cohere.Client(cohere.api_key)
    return cohere_agent


def get_huggingface_headers():
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"
    }
    return HUGGINGFACE_HEADERS


def _value_is_set(v: str) -> bool:
    return not(v is None or "fill" in v or v == "")


class ApiKeyError(RuntimeError):
    def __init__(self, *args, key: str, msg: str = ""):
        super().__init__(msg, *args)
        self.key = key
        self.msg = msg


def _check_key(k: str, v: str = None) -> None:
    """
    Check that the given `k` is an env var with a value that indicates a valid
    api key or secret.  If `v` is provided, checks that instead. If value
    indicates the key is not set, raises an informative error telling the user
    options on how to set that key.
    """

    v = v or os.environ.get(k)

    if not _value_is_set(v):
        msg = f"""Key {k} needs to be set; please provide it in one of these ways:

  - in a variable {k} prior to this check, 
  - in your variable environment, 
  - in a .env file in {Path.cwd()} or its parents,
  - explicitly passed to function `check_or_set_keys` of `trulens_eval.keys`,
  - passed to the endpoint or feedback collection constructor that needs it (`trulens_eval.provider_apis.OpenAIEndpoint`, etc.), or
  - set in api utility class that expects it (i.e. `openai`, etc.).

For the last two options, the name of the argument may differ from {k} (i.e. `openai.api_key` for `OPENAI_API_KEY`).
"""
        print(f"{UNICODE_STOP} {msg}")
        raise ApiKeyError(key=k, msg=msg)


def _relative_path(path: Path, relative_to: Path) -> str:
    """
    Get the path `path` relative to path `relative_to` even if `relative_to` is
    not a prefix of `path`. Iteratively takes the parent of `relative_to` in
    that case until it becomes a prefix. Each parent is indicated by '..'.
    """

    parents = 0

    while True:
        try:
            return "".join(["../"] * parents) + str(path.relative_to(relative_to))
        except Exception:
            parents += 1
            relative_to = relative_to.parent
    

def _collect_keys(*args, **kwargs) -> dict:
    """
    Collect values for keys from all of the currently supported sources. This includes:

    - Using env variables.

    - Using python variables.

    - Explicitly passed to `check_or_set_keys`.

    - Using vars defined in a .env file in current folder or one of its parents.

    - Using 3rd party class attributes (i.e. OpenAI.api_key). This one requires the
      user to initialize our Endpoint class for that 3rd party api.

    - With initialization of trulens_eval Endpoint class that handles a 3rd party api.
    """

    ret = dict()

    config_file, config = get_config()
    
    globs = caller_frame(offset=2).f_globals

    for k in list(args) + list(kwargs.keys()):
        valid_values = set()
        valid_sources = defaultdict(list)

        # Env vars. NOTE: Endpoint classes copy over relevant keys from 3rd party
        # classes (or provided explicitly to them) to var env.
        temp_v = os.environ.get(k)
        if _value_is_set(temp_v):
            valid_sources[temp_v].append("environment")
            valid_values.add(temp_v)

        # Explicit.
        temp_v = kwargs.get(k)
        if _value_is_set(temp_v):
            valid_sources[temp_v].append(f"explicit value to `check_or_set_keys`")
            valid_values.add(temp_v)

        
        # .env vars.
        if config is not None:
            temp_v = config.get(k)
            if _value_is_set(temp_v):
                valid_sources[temp_v].append(f".env file at {config_file}")
                valid_values.add(temp_v)

        # Globals of caller.
        temp_v = globs.get(k)
        if _value_is_set(temp_v):
            valid_sources[temp_v].append(f"python variable")
            valid_values.add(temp_v)

        if len(valid_values) == 0:
            ret[k] = None

        elif len(valid_values) > 1:
            warning = f"More than one different value for key {k} has been found:\n\t"
            warning += "\n\t".join(f"""value ending in {v[-1]} in {' and '.join(valid_sources[v])}""" for v in valid_values)
            warning += f"\nUsing one arbitrarily."
            logger.warning(warning)

            ret[k] = list(valid_values)[0]
        else:
            v = list(valid_values)[0]
            print(
                f"{UNICODE_CHECK} Key {k} set from {valid_sources[v][0]}"
                + (' (same value found in ' + (' and '.join(valid_sources[v][1:])) + ')' if len(valid_sources[v]) > 1 else '')
                + "."
            )

            ret[k] = v
        
    return ret


def check_keys(*keys):
    """
    Check that all keys named in `*args` are set as env vars. Will fail with a
    message on how to set missing key if one is missing. If all are provided
    somewhere, they will be set in the env var as the canonical location where
    we should expect them subsequently. Example:

    ```python 
    from trulens_eval.keys import check_keys

    check_keys(
        "OPENAI_API_KEY",
        "HUGGINGFACE_API_KEY"
    )
    ```
    """

    kvals = _collect_keys(*keys)
    for k in keys:
        v = kvals.get(k)
        _check_key(k, v=v)
        os.environ[k] = v


def check_or_set_keys(*args, **kwargs):
    """
    Check various sources of api configuration values like secret keys and set
    env variables for each of them. We use env variables as the canonical
    storage of these keys, regardless of how they were specified. Values can
    also be specified explicitly to this method. Example:

    ```python 
    from trulens_eval.keys import check_or_set_keys

    check_or_set_keys(
        OPENAI_API_KEY="to fill in", 
        HUGGINGFACE_API_KEY="to fill in"
    )
    ```
    """

    kvals = _collect_keys(*args, **kwargs)
    for k in list(args) + list(kwargs.keys()):
        v = kvals.get(k)
        _check_key(k, v=v)
        os.environ[k] = v

