"""Import utilities for required and optional imports.

Utilities for importing python modules and optional importing.
"""

import builtins
from dataclasses import dataclass
import importlib
from importlib import metadata
from importlib import resources
import inspect
from inspect import cleandoc
import logging
from pathlib import Path
from pprint import PrettyPrinter
import subprocess
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from packaging import requirements
from packaging import version
from pip._internal.req import parse_requirements
from trulens.core.utils.python import caller_frame
from trulens.core.utils.python import caller_module

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

NO_INSTALL: bool = False
"""If set, will not automatically install any optional trulens modules and fail
in the typical way if a module is missing."""


def pip(*args) -> subprocess.CompletedProcess:
    """Execute pip with the given arguments."""

    logger.debug("Running pip", *args)

    return subprocess.run(
        [sys.executable, "-m", "pip", *args], capture_output=True, check=True
    )


def install(package_name: str):
    """Install a package with pip."""

    package_name = safe_importlib_package_name(package_name)

    logger.info("Installing package %s", package_name)

    pip("install", package_name)

    return get_package_version(package_name)


def make_help_str(
    kinds: Dict[str, Dict],
) -> Tuple[Callable[[], None], Callable[[], str]]:
    """Create a help string that lists all available classes and their installation status."""

    mod = sys.modules[caller_frame(offset=1).f_locals["__name__"]]

    def help_str() -> str:
        ret = f"Module '{mod.__name__}' from '{mod.__file__} contains:\n"
        for kind, items in kinds.items():
            ret += f"{kind}: \n"  # TODO: pluralize
            for class_, (package_name, _) in items.items():
                version = get_package_version(package_name)
                ret += (
                    f"  {class_}\t[{package_name}]\t["
                    + (str(version) if version is not None else "not installed")
                    + "]\n"
                )

        if NO_INSTALL:
            ret += "\nYou can enable automatic installs by calling `trulens.api.set_no_install(False)`."
        else:
            ret += "\nImporting from this module will install the required package. You can disable this by calling `trulens.api.set_no_install()`."

        return ret

    def help() -> None:
        """Print a help message that lists all available classes and their installation status."""

        print(help_str())

    return help, help_str


def make_getattr_override(
    # mod: ModuleType,
    kinds: Dict[str, Dict],
    help_str: Union[str, Callable[[], str]],
) -> Any:
    """Make a custom __getattr__ function for a module to allow automatic
    installs of missing modules and better error messages."""

    mod = sys.modules[caller_module(offset=1)]

    def getattr_(attr):
        nonlocal mod

        if attr in [
            "_ipython_canary_method_should_not_exist_",
            "_ipython_display_",
            "__warningregistry__",
        ]:
            raise AttributeError()

        for kind, options in kinds.items():
            if attr in options:
                package_name, module_name = options[attr]

                try:
                    mod = import_module(
                        package_name=package_name,
                        module_name=module_name,
                        purpose=f"using the {attr} {kind}",
                    )
                    return getattr(mod, attr)

                except ImportError as e:
                    raise ImportError(
                        f"""Could not import the {attr} {kind}. You might need to re-install {package_name}:
            ```bash
            pip uninstall -y {package_name}
            pip install {package_name}
            ```
        """
                    ) from e

        # We also need to the case of an import of a sub-module instead of
        # something in __all__ (enumerated in kinds) so we try that here.
        try:
            submod = importlib.import_module(mod.__name__ + "." + attr)
            return submod
        except Exception:
            # Use the same error message as the below.
            pass

        raise AttributeError(
            f"Module {mod.__name__} has no attribute {attr}.\n"
            + (help_str if isinstance(help_str, str) else help_str())
        )

    return getattr_


def import_module(
    package_name: str, module_name: str, purpose: Optional[str] = None
):
    """Import a module of the given package but install it if it is not already
    installed.

    If purpose is given, will use that for the message printed before the
    installation if it does happen.
    """

    package_name = safe_importlib_package_name(package_name)

    if purpose is None:
        purpose = f"using {module_name}"

    try:
        if (package := get_package_version(package_name)) is None:
            if NO_INSTALL:
                # Throwing RuntimeError as I don't want this to be caught by the
                # importing try blocks. Needs this to be the final error.
                raise RuntimeError(
                    f"Package {package_name} is not installed. Enable automatic installation by calling `trulens.api.set_no_install(False)` or install it manually with pip: \n\t```bash\n\tpip install '{package_name}'\n\t```"
                )

            if sys.version_info >= (3, 12) and package_name in [
                "trulens-providers-cortex",
                "trulens-instrument-nemo",
            ]:
                # These require python < 3.12 .
                # RuntimeError here on purpose. See above.
                raise RuntimeError(
                    f"Package {package_name} required for {purpose} does not support python >= 3.12."
                )

            else:
                print(
                    f"Installing package {package_name} required for {purpose} ..."
                )

            package = install(package_name)

    except subprocess.CalledProcessError as e:
        # ImportError here on purpose. Want this to be caught in importing try
        # blocks to give additional information.
        raise ImportError(f"Could not install {package_name}.") from e

    if package is None:
        # Same
        raise ImportError(f"Could not install {package_name}.")

    return importlib.import_module(module_name)


def safe_importlib_package_name(package_name: str) -> str:
    """Convert a package name that may have periods in it to one that uses
    hyphens for periods but only if the python version is old."""

    return (
        package_name
        if sys.version_info >= (3, 10)
        else package_name.replace(".", "-")
    )


def requirements_of_file(path: Path) -> Dict[str, requirements.Requirement]:
    """Get a dictionary of package names to requirements from a requirements
    file."""

    pip_reqs = parse_requirements(str(path), session=None)

    mapping = {}

    for pip_req in pip_reqs:
        req = requirements.Requirement(pip_req.requirement)
        mapping[req.name] = req

    return mapping


def static_resource(namespace: str, filepath: Union[Path, str]) -> Path:
    """Get the path to a static resource file in the trulens package.

    By static here we mean something that exists in the filesystem already and
    not in some temporary folder. We use the `importlib.resources` context
    managers to get this but if the resource is temporary, the result might not
    exist by the time we return or is not expected to survive long.
    """

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if sys.version_info >= (3, 9):
        # This does not exist in 3.8
        from importlib.abc import Traversable

        _trulens_resources: Traversable = resources.files(
            f"trulens.{namespace}"
        )
        with resources.as_file(_trulens_resources / filepath) as _path:
            return _path
    else:
        # This is deprecated starting 3.11
        parts = filepath.parts
        with resources.path("trulens", parts[0]) as _path:
            # NOTE: resources.path does not allow the resource to include folders.
            for part in parts[1:]:
                _path = _path / part
            return _path


def parse_version(version_string: str) -> version.Version:
    """Parse the version string into a packaging version object."""

    return version.parse(version_string)


def get_package_version(name: str) -> Optional[version.Version]:
    """Get the version of a package by its name.

    Returns None if given package is not installed.
    """

    try:
        return parse_version(metadata.version(name))

    except metadata.PackageNotFoundError:
        return None


MESSAGE_DEBUG_OPTIONAL_PACKAGE_NOT_FOUND = """Optional package %s is not installed. Related optional functionality will not
be available.
"""

MESSAGE_ERROR_REQUIRED_PACKAGE_NOT_FOUND = """Required package {req.name} is not installed. Please install it with pip:

    ```bash
    pip install '{req}'
    ```

If your distribution is in a bad place beyond this package, you may need to
reinstall trulens so that all of the dependencies get installed:

    ```bash
    pip uninstall -y trulens
    pip install trulens
    ```
"""

MESSAGE_FRAGMENT_VERSION_MISMATCH = """Package {req.name} is installed but has a version conflict:
    Requirement: {req}
    Installed: {dist.version}
"""

MESSAGE_FRAGMENT_VERSION_MISMATCH_OPTIONAL = """This package is optional for trulens so this may not be a problem but if
you need to use the related optional features and find there are errors, you
will need to resolve the conflict:
"""

MESSAGE_FRAGMENT_VERSION_MISMATCH_REQUIRED = """This package is required for trulens. Please resolve the conflict by
installing a compatible version with:
"""

MESSAGE_FRAGMENT_VERSION_MISMATCH_PIP = """
    ```bash
    pip install '{req}'
    ```

If you are running trulens in a notebook, you may need to restart the
kernel after resolving the conflict. If your distribution is in a bad place
beyond this package, you may need to reinstall trulens so that all of the
dependencies get installed and hopefully corrected:

    ```bash
    pip uninstall -y trulens
    pip install trulens
    ```
"""


def pin_spec(r: requirements.Requirement) -> requirements.Requirement:
    """
    Pin the requirement to the version assuming it is lower bounded by a
    version.
    """

    spec = str(r)
    if ">=" not in spec:
        raise ValueError(f"Requirement {spec} is not lower-bounded.")

    spec = spec.replace(">=", "==")
    return requirements.Requirement(spec)


@dataclass
class ImportErrorMessages:
    """Container for messages to show when an optional package is not found or
    has some other import error."""

    module_not_found: str
    """Message to show or raise when a package is not found."""

    import_error: str
    """Message to show or raise when a package may be installed but some import
    error occurred trying to import it or something from it."""


def format_import_errors(
    packages: Union[str, Sequence[str]],
    purpose: Optional[str] = None,
    throw: Union[bool, Exception] = False,
) -> ImportErrorMessages:
    """Format two messages for missing optional package or bad import from an
    optional package.

    Throws an `ImportError` with the formatted message if `throw` flag is set.
    If `throw` is already an exception, throws that instead after printing the
    message.
    """

    if purpose is None:
        purpose = f"using {packages}"

    if isinstance(packages, str):
        packages = [packages]

    requirements = []
    requirements_pinned = []

    packs = ",".join(packages)
    pack_s = "package" if len(packages) == 1 else "packages"
    is_are = "is" if len(packages) == 1 else "are"
    it_them = "it" if len(packages) == 1 else "them"
    this_these = "this" if len(packages) == 1 else "these"

    msg = cleandoc(f"""
{",".join(packages)} {pack_s} {is_are} required for {purpose}.
You should be able to install {it_them} with pip:

    ```bash
    pip install {" ".join(map(lambda a: f'"{a}"', requirements))}
    ```
""")

    msg_pinned = cleandoc(f"""
You have {packs} installed but we could not import the required
components. There may be a version incompatibility. Please try installing {this_these}
exact {pack_s} with pip:

    ```bash
    pip install {" ".join(map(lambda a: f'"{a}"', requirements_pinned))}
    ```

Alternatively, if you do not need {packs}, uninstall {it_them}:

    ```bash
    pip uninstall -y '{" ".join(packages)}'
    ```
""")

    if isinstance(throw, Exception):
        print(msg)
        raise throw

    elif isinstance(throw, bool):
        if throw:
            raise ImportError(msg)

    return ImportErrorMessages(module_not_found=msg, import_error=msg_pinned)


# To remove after trulens_eval is removed:

# Optional sub-packages:
REQUIREMENT_FEEDBACK = format_import_errors(
    "trulens-feedback", purpose="evaluating feedback functions"
)

# Optional app types:
REQUIREMENT_INSTRUMENT_LLAMA = format_import_errors(
    "trulens-instrument-llamaindex", purpose="instrumenting LlamaIndex apps"
)
REQUIREMENT_INSTRUMENT_LANGCHAIN = format_import_errors(
    "trulens-instrument-langchain", purpose="instrumenting LangChain apps"
)
REQUIREMENT_INSTRUMENT_NEMO = format_import_errors(
    "trulens-instrument-nemo", purpose="instrumenting NeMo Guardrails apps"
)

# Optional provider types:

REQUIREMENT_PROVIDER_BEDROCK = format_import_errors(
    "trulens-providers-bedrock", purpose="evaluating feedback using Bedrock"
)
REQUIREMENT_PROVIDER_CORTEX = format_import_errors(
    "trulens-providers-cortex", purpose="evaluating feedback using Cortex"
)
REQUIREMENT_PROVIDER_HUGGINGFACE = format_import_errors(
    "trulens-providers-huggingface",
    purpose="evaluating feedback using Huggingface",
)
REQUIREMENT_PROVIDER_LANGCHAIN = format_import_errors(
    "trulens-providers-langchain", purpose="evaluating feedback using LangChain"
)
REQUIREMENT_PROVIDER_LITELLM = format_import_errors(
    "trulens-providers-litellm", purpose="evaluating feedback using LiteLLM"
)
REQUIREMENT_PROVIDER_OPENAI = format_import_errors(
    "trulens-providers-openai", purpose="evaluating feedback using OpenAI"
)

# Other optionals:
REQUIREMENT_SKLEARN = format_import_errors(
    "scikit-learn", purpose="using embedding vector distances"
)
REQUIREMENT_BERT_SCORE = format_import_errors(
    "bert-score", purpose="measuring BERT Score"
)
REQUIREMENT_EVALUATE = format_import_errors(
    "evaluate", purpose="using certain metrics"
)
REQUIREMENT_NOTEBOOK = format_import_errors(
    ["ipython", "ipywidgets"], purpose="using TruLens-Eval in a notebook"
)
REQUIREMENT_OPENAI = format_import_errors(
    ["openai", "langchain_community"], purpose="using OpenAI models"
)

REQUIREMENT_SNOWFLAKE = format_import_errors(
    [
        "snowflake-core",
        "snowflake-connector-python",
        "snowflake-snowpark-python",
        "snowflake-sqlalchemy",
    ],
    purpose="connecting to Snowflake",
)


# Try to pretend to be a type as well as an instance.
class Dummy(type):
    """Class to pretend to be a module or some other imported object.

    Will raise an error if accessed in some dynamic way. Accesses that are
    "static-ish" will try not to raise the exception so things like defining
    subclasses of a missing class should not raise exception. Dynamic uses are
    things like calls, use in expressions. Looking up an attribute is static-ish
    so we don't throw the error at that point but instead make more dummies.


    Warning:
        While dummies can be used as types, they return false to all `isinstance`
        and `issubclass` checks. Further, the use of a dummy in subclassing
        produces unreliable results with some of the debugging information such
        as `original_exception` may be inaccassible.
    """

    def __str__(self) -> str:
        ret = f"Dummy({self.name}"

        if self.original_exception is not None:
            ret += f", original_exception={self.original_exception}"

        ret += ")"

        return ret

    def __repr__(self) -> str:
        return str(self)

    def __new__(cls, name, *args, **kwargs):
        if (
            len(args) >= 2
            and isinstance(args[1], dict)
            and "__classcell__" in args[1]
        ):
            # Used as type, for subclassing for example.

            return type.__new__(cls, name, args[0], args[1])
        else:
            return type.__new__(cls, name, (cls,), kwargs)

    def __init__(self, name: str, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[1], dict):
            # Used as type, in subclassing for example.

            src = args[0][0]

            message = src.message
            importer = src.importer
            original_exception = src.original_exception
            exception_class = src.exception_class

        else:
            message: str = kwargs.get("message", None)
            exception_class: Type[Exception] = kwargs.get(
                "exception_class", ModuleNotFoundError
            )
            importer = kwargs.get("importer", None)
            original_exception: Optional[Exception] = kwargs.get(
                "original_exception", None
            )

        self.name = name
        self.message = message
        self.importer = importer
        self.exception_class = exception_class
        self.original_exception = original_exception

    def __call__(self, *args, **kwargs):
        raise self.exception_class(self.message) from self.original_exception

    def __instancecheck__(self, __instance: Any) -> bool:
        """Nothing is an instance of this dummy.

        Warning:
            This is to make sure that if something optional gets imported as a
            dummy and is a class to be instrumented, it will not automatically make
            the instrumentation class check succeed on all objects.
        """
        return False

    def __subclasscheck__(self, __subclass: type) -> bool:
        """Nothing is a subclass of this dummy."""

        return False

    def _wasused(self, *args, **kwargs):
        raise self.exception_class(self.message) from self.original_exception

    # If someone tries to use dummy in an expression, raise our usage exception:
    __add__ = _wasused
    __sub__ = _wasused
    __mul__ = _wasused
    __truediv__ = _wasused
    __floordiv__ = _wasused
    __mod__ = _wasused
    __divmod__ = _wasused
    __pow__ = _wasused
    __lshift__ = _wasused
    __rshift__ = _wasused
    __and__ = _wasused
    __xor__ = _wasused
    __or__ = _wasused
    __radd__ = _wasused
    __rsub__ = _wasused

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested attribute. Otherwise
        # raise error.

        # Pretend to be object for generic attributes.
        if hasattr(object, name):
            return getattr(object, name)

        # Prevent pydantic inspecting this object as if it were a type from triggering the exception
        # message below.
        if name in [
            "__pydantic_generic_metadata__",
            "__get_pydantic_core_schema__",
            "__get_validators__",
            "__get_pydantic_json_schema__",
            "__modify_schema__",
            "__origin__",
            "__dataclass_fields__",
        ]:
            raise AttributeError()

        # If we are still in an optional import block, continue making dummies
        # inside this dummy.
        if self.importer is not None and (
            self.importer.importing and not self.importer.fail
        ):
            return Dummy(
                name=self.name + "." + name,
                message=self.message,
                importer=self.importer,
                original_exception=self.original_exception,
                exception_class=ModuleNotFoundError,
            )

        # If we are no longer in optional imports context or context said to
        # fail anyway, raise the exception with the optional package message.

        raise self.exception_class(self.message)


class OptionalImports:
    """Helper context manager for doing multiple imports from an optional
    modules

    Example:
        ```python
            messages = ImportErrorMessages(
                module_not_found="install llama_index first",
                import_error="install llama_index==0.1.0"
            )
            with OptionalImports(messages=messages):
                import llama_index
                from llama_index import query_engine
        ```

        The above python block will not raise any errors but once anything else
        about llama_index or query_engine gets accessed, an error is raised with the
        specified message (unless llama_index is installed of course).
    """

    def assert_installed(self, mods: Union[Any, Iterable[Any]]):
        """Check that the given modules `mods` are not dummies. If any is, show the
        optional requirement message.

        Returns self for chaining convenience.
        """
        if not isinstance(mods, Iterable):
            mods = [mods]

        if any(isinstance(mod, Dummy) for mod in mods):
            raise ModuleNotFoundError(self.messages.module_not_found)

        return self

    def __init__(self, messages: ImportErrorMessages, fail: bool = False):
        """
        Create an optional imports context manager class. Will keep module not
        found or import errors quiet inside context unless fail is True.
        """

        self.messages = messages
        self.importing = False
        self.fail = fail
        self.imp = builtins.__import__

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        # Check if this import call is coming from an import in trulens as
        # otherwise we don't want to intercept the error as some modules rely on
        # import failures for various things. HACK012: we have to step back a
        # frame or two here to check where the original import came from. We
        # skip any frames that refer to our overwritten __import__. We have to
        # step back multiple times if we (accidentally) nest our OptionalImport
        # context manager.
        frame = inspect.currentframe().f_back
        while frame.f_code == self.__import__.__code__:
            frame = frame.f_back

        module_name = frame.f_globals["__name__"]

        if not module_name.startswith("trulens"):
            return self.imp(name, globals, locals, fromlist, level)

        try:
            mod = self.imp(name, globals, locals, fromlist, level)

            # NOTE(piotrm): commented block attempts to check module contents for required
            # attributes so we can offer a message without raising an exception later. It is
            # commented out for now it is catching some things we don't watch to catch. Need to
            # check how attributes are normally looked up in a module to fix this. Specifically, the
            # code that raises these messages: "ImportError: cannot import name ..."
            """
            if isinstance(fromlist, Iterable):
                for i in fromlist:
                    if i == "*":
                        continue
                    # Check the contents so there is opportunity to catch import errors here
                    try:
                        getattr(mod, i)
                    except AttributeError as e:
                        raise ImportError(e)
            """

            return mod

        except ModuleNotFoundError as e:
            if self.fail:
                raise e

            return Dummy(
                name=name,
                message=self.messages.module_not_found,
                importer=self,
                original_exception=e,
                exception_class=ModuleNotFoundError,
            )

        except AttributeError as e:
            if self.fail:
                raise e

            return Dummy(
                name=name,
                message=self.messages.module_not_found,
                importer=self,
                original_exception=e,
                exception_class=AttributeError,
            )

        except ImportError as e:
            if self.fail:
                raise e

            return Dummy(
                name=name,
                message=self.messages.import_error,
                exception_class=ImportError,
                importer=self,
                original_exception=e,
            )

    def __enter__(self):
        """Handle entering the WithOptionalImports context block.

        We override the builtins.__import__ function to catch any import errors.
        """

        # TODO: Better handle nested contexts. For now we don't override import
        # and just let the already-overridden one do its thing.

        if "trulens" in str(builtins.__import__):
            logger.debug(
                "Nested optional imports context used. This context will be ignored."
            )
        else:
            builtins.__import__ = self.__import__

        self.importing = True

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Handle exiting from the WithOptionalImports context block.

        We should not get any exceptions here if dummies were produced by the
        overwritten __import__ but if an import of a module that exists failed
        becomes some component of that module did not, we will not be able to
        catch it to produce dummy and have to process the exception here in
        which case we add our informative message to the exception and re-raise
        it.
        """
        self.importing = False
        builtins.__import__ = self.imp

        if exc_value is None:
            return

        if isinstance(exc_value, ModuleNotFoundError):
            if exc_value.msg.startswith(self.messages.module_not_found):
                # Don't add more to the message if it already includes our instructions.
                raise exc_value

            raise ModuleNotFoundError(
                self.messages.module_not_found
            ) from exc_value

        elif isinstance(exc_value, ImportError):
            if exc_value.msg.startswith(self.messages.import_error):
                # Don't add more to the message if it already includes our instructions.
                raise exc_value

            raise ImportError(self.messages.import_error) from exc_value

        # Exception will be propagated unless we return True so we don't return it.
