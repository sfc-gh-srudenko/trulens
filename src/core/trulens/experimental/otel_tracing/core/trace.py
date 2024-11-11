# ruff: noqa: E402

"""Implementation of recording that resembles the tracing process in OpenTelemetry.

!!! Note
    Most of the module is EXPERIMENTAL(otel_tracing) though it includes some existing
    non-experimental classes moved here to resolve some circular import issues.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import os
import threading as th
from threading import Lock
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import uuid
import weakref

import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing import _feature
from trulens.experimental.otel_tracing.core import otel as core_otel
from trulens.experimental.otel_tracing.core._utils import wrap as wrap_utils
from trulens.semconv import trace as truconv

_feature._FeatureSetup.assert_optionals_installed()  # checks to make sure otel is installed

from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import span as span_api
from opentelemetry.util import types as types_api

T = TypeVar("T")
R = TypeVar("R")  # callable return type
E = TypeVar("E")  # iterator/generator element type

logger = logging.getLogger(__name__)

INSTRUMENT: str = "__tru_instrumented"
"""Attribute name to be used to flag instrumented objects/methods/others."""

APPS: str = "__tru_apps"
"""Attribute name for storing apps that expect to be notified of calls."""


class SpanContext(core_otel.SpanContext, Hashable):
    """TruLens additions on top of OTEL SpanContext to add Hashable and
    reference to tracer that made the span."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        return f"{self.trace_id % 0xFF:02x}/{self.span_id % 0xFF:02x}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.trace_id + self.span_id

    def __eq__(self, other: SpanContextLike):
        if other is None:
            return False

        return self.trace_id == other.trace_id and self.span_id == other.span_id

    @staticmethod
    def of_spancontextlike(span_context: SpanContextLike) -> SpanContext:
        if isinstance(span_context, SpanContext):
            return span_context

        elif isinstance(span_context, core_otel.SpanContext):
            return SpanContext(
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                is_remote=span_context.is_remote,
            )
        elif isinstance(span_context, span_api.SpanContext):
            return SpanContext(
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                is_remote=span_context.is_remote,
            )
        elif isinstance(span_context, Dict):
            return SpanContext.model_validate(span_context)
        else:
            raise ValueError(f"Unrecognized span context type: {span_context}")


SpanContextLike = Union[
    SpanContext, core_otel.SpanContext, span_api.SpanContext, serial_utils.JSON
]
"""SpanContext types we need to deal with.

These may be the non-hashable ones coming from OTEL, the hashable ones we
create, or their JSON representations."""


class Span(core_otel.Span):
    """TruLens additions on top of OTEL spans.

    Note that in this representation, we keep track of the tracer that produced
    the instance and have properties to access other spans from that tracer,
    like the parent. This make traversing lives produced in this process a bit
    easier.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # model_validate will fail without this
    )

    trace_record_ids: Dict[types_schema.AppID, types_schema.RecordID] = (
        pydantic.Field(default_factory=dict)
    )
    """Id of the record this span belongs to, per app.

    This is because the same span might represent part of the trace of different
    records because more than one app is tracing.

    This will not be filled in if the span was produced outside of a recording
    context.
    """

    def __str__(self):
        return (
            f"{type(self).__name__}({self.name}, {self.context}->{self.parent})"
        )

    def __repr__(self):
        return str(self)

    _lensed_attributes: serial_utils.LensedDict[Any] = pydantic.PrivateAttr(
        default_factory=serial_utils.LensedDict
    )

    @property
    def lensed_attributes(self) -> serial_utils.LensedDict[Any]:
        return self._lensed_attributes

    @property
    def parent_span(self) -> Optional[Span]:
        if self.parent is None:
            return None

        if self._tracer is None:
            return None

        if (span := self._tracer.spans.get(self.parent)) is None:
            return None

        return span

    _children_spans: List[Span] = pydantic.PrivateAttr(default_factory=list)

    @property
    def children_spans(self) -> List[Span]:
        return self._children_spans

    error: Optional[Exception] = pydantic.Field(None)
    """Optional error if the observed computation raised an exception."""

    def __init__(self, **kwargs):
        # Convert any contexts to our hashable context class:
        if (context := kwargs.get("context")) is not None:
            kwargs["context"] = SpanContext.of_spancontextlike(context)
        if (parent := kwargs.get("parent", None)) is not None:
            kwargs["parent"] = SpanContext.of_spancontextlike(parent)

        super().__init__(**kwargs)

        if (parent_span := self.parent_span) is not None:
            parent_span.children_spans.append(self)

    def iter_ancestors(self) -> Iterable[Span]:
        """Iterate over all ancestors of this span."""

        yield self

        if self.parent_span is not None:
            yield from self.parent_span.iter_ancestors()

    def has_ancestor_of_type(self, span_type: Type[Span]) -> bool:
        """Check if this span has an ancestor of the given type."""

        for ancestor in self.iter_ancestors():
            if isinstance(ancestor, span_type):
                return True

        return False

    def iter_children(
        self, transitive: bool = True, include_phantom: bool = False
    ) -> Iterable[Span]:
        """Iterate over all spans that are children of this span.

        Args:
            transitive: Iterate recursively over children.

            include_phantom: Include phantom spans. If not set, phantom spans
                will not be included but will be iterated over even if
                transitive is false.
        """

        for child_span in self.children_spans:
            if isinstance(child_span, PhantomSpan) and not include_phantom:
                # Note that transitive being false is ignored if phantom is skipped.
                yield from child_span.iter_children(
                    transitive=transitive, include_phantom=include_phantom
                )
            else:
                yield child_span
                if transitive:
                    yield from child_span.iter_children(
                        transitive=transitive,
                        include_phantom=include_phantom,
                    )

    def iter_family(self, include_phantom: bool = False) -> Iterable[Span]:
        """Iterate itself and all children transitively."""

        if (not isinstance(self, PhantomSpan)) or include_phantom:
            yield self

        yield from self.iter_children(
            include_phantom=include_phantom, transitive=True
        )

    def total_cost(self) -> base_schema.Cost:
        """Total costs of this span and all its transitive children."""

        total = base_schema.Cost()

        for span in self.iter_family(include_phantom=True):
            if isinstance(span, WithCost) and span.cost is not None:
                total += span.cost

        return total


class PhantomSpan(Span):
    """A span type that indicates that it does not correspond to a
    computation to be recorded but instead is an element of the tracing system.

    It is to be removed from the spans presented to the users or exported.
    """


class LiveSpan(Span):
    """A a span type that indicates that it contains live python objects.

    It is to be converted to a non-live span before being output to the user or
    otherwise.
    """

    live_apps: weakref.WeakSet[Any] = pydantic.Field(
        default_factory=weakref.WeakSet, exclude=True
    )  # Any = App
    """Apps for which this span is recording trace info for.

    WeakSet to prevent memory leaks.

    Note that this will not be filled in if this span was produced outside of an
    app recording context.
    """


class PhantomSpanRecordingContext(PhantomSpan):
    """Tracks the context of an app used as a context manager."""

    recording: Optional[Any] = pydantic.Field(None, exclude=True)
    # TODO: app.RecordingContext # circular import issues

    def otel_resource_attributes(self) -> Dict[str, Any]:
        ret = super().otel_resource_attributes()

        ret[ResourceAttributes.SERVICE_NAME] = (
            self.recording.app.app_name if self.recording is not None else None
        )

        return ret

    # override
    def end(self, *args, **kwargs):
        super().end(*args, **kwargs)

        self._finalize_recording()

    # override
    def record_exception(
        self,
        exception: BaseException,
        attributes: types_api.Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        super().record_exception(exception, attributes, timestamp, escaped)

        self._finalize_recording()

    def _finalize_recording(self):
        assert self.recording is not None

        app = self.recording.app

        for span in Tracer.find_each_child(
            span=self, span_filter=lambda s: isinstance(s, LiveRecordRoot)
        ):
            app._on_new_root_span(recording=self.recording, root_span=span)

        app._on_new_recording_span(recording_span=self)

    def otel_name(self) -> str:
        return "trulens.recording"


class LiveSpanCall(LiveSpan):
    """Track a function call."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    call_id: Optional[uuid.UUID] = pydantic.Field(None)
    """Unique call identifiers."""

    process_id: Optional[int] = pydantic.Field(None, exclude=True)
    """Process ID of the call."""

    thread_id: Optional[int] = pydantic.Field(None, exclude=True)
    """Thread ID of the call."""

    live_sig: Optional[inspect.Signature] = pydantic.Field(None, exclude=True)
    """Called function's signature."""

    live_obj: Optional[Any] = pydantic.Field(None, exclude=True)
    """Self object if method call."""

    live_cls: Optional[Type] = pydantic.Field(None, exclude=True)
    """Class if method/static/class method call."""

    live_func: Optional[Callable] = pydantic.Field(None, exclude=True)
    """Function object."""

    live_args: Optional[Tuple[Any, ...]] = pydantic.Field(None, exclude=True)
    """Positional arguments to the function call."""

    live_kwargs: Optional[Dict[str, Any]] = pydantic.Field(None, exclude=True)
    """Keyword arguments to the function call."""

    live_bindings: Optional[inspect.BoundArguments] = pydantic.Field(
        None, exclude=True
    )
    """Bound arguments to the function call if can be bound."""

    live_ret: Optional[Any] = pydantic.Field(None, exclude=True)
    """Return value of the function call.

    Exclusive with `error`.
    """

    live_error: Optional[Any] = pydantic.Field(None, exclude=True)
    """Error raised by the function call.

    Exclusive with `ret`.
    """


class LiveRecordRoot(LiveSpan):
    """Wrapper for first app calls, or "records".

    Children spans of type `WithApps` are expected to contain the app named here
    in their `live_apps` field and have a record_id for this app.
    """

    live_app: Optional[weakref.ReferenceType[Any]] = pydantic.Field(
        None, exclude=True
    )  # Any = App
    """The app for which this is the root call.

    Value must be included in childrens' `live_apps` field.
    """

    trace_record_id: types_schema.TraceRecordID.PY_TYPE = pydantic.Field(
        default_factory=types_schema.TraceRecordID.default_py
    )
    """Unique identifier for this root call or what is called a "record".

    Note that this is different from `trace_record_ids` though this
    `trace_record_id` will be included in `record_ids` and will be included in
    children's `trace_record_ids` fields.

    Note that a record root cannot be a distributed call hence there is no
    non-live record root.
    """


S = TypeVar("S", bound=LiveSpanCall)


class WithCost(LiveSpan):
    """Mixin to indicate the span has costs tracked."""

    cost: base_schema.Cost = pydantic.Field(default_factory=base_schema.Cost)
    """Cost of the computation spanned."""

    endpoint: Optional[Any] = pydantic.Field(
        None, exclude=True
    )  # Any actually core_endpoint.Endpoint
    """Endpoint handling cost extraction for this span/call."""

    def __init__(self, cost: Optional[base_schema.Cost] = None, **kwargs):
        if cost is None:
            cost = base_schema.Cost()

        super().__init__(cost=cost, **kwargs)


class LiveSpanCallWithCost(LiveSpanCall, WithCost):
    pass


class Tracer(core_otel.Tracer):
    """TruLens additions on top of [OTEL Tracer][opentelemetry.trace.Tracer]."""

    # TODO: Create a Tracer that does not record anything. Can either be a
    # setting to this tracer or a separate "NullTracer". We need non-recording
    # users to not incur much overhead hence need to be able to disable most of
    # the tracing logic when appropriate.

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Overrides core_otel.Tracer._span_class
    _span_class: Type[Span] = pydantic.PrivateAttr(Span)

    # Overrides core_otel.Tracer._span_context_class
    _span_context_class: Type[SpanContext] = pydantic.PrivateAttr(SpanContext)

    @property
    def spans(self) -> Dict[SpanContext, Span]:
        return self._tracer_provider.spans

    @property
    def current_span(self) -> Optional[Span]:
        if (context := self.span_context) is None:
            return None

        return self.spans.get(context)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return (
            type(self).__name__
            + " "
            + self.instrumenting_module_name
            + " "
            + self.instrumenting_library_version
        )

    def __repr__(self):
        return str(self)

    def start_span(self, *args, **kwargs):
        """Like OTEL start_span except also keeps track of the span just created."""

        new_span = super().start_span(*args, **kwargs)

        self.spans[new_span.context] = new_span

        return new_span

    @staticmethod
    def _fill_stacks(
        span: Span,
        get_method_path: Callable,
        span_stacks: Dict[Span, List[record_schema.RecordAppCallMethod]],
        stack: Optional[List[record_schema.RecordAppCallMethod]] = None,
    ):
        """Populate span_stacks with a mapping of span to call stack for
        backwards compatibility with records.

        Args:
            span: Span to start from.

            get_method_path: Function that looks up lens of a given
                obj/function. This is an WithAppCallbacks method.

            span_stacks: Mapping of span to call stack. This will be modified by
                this method.

            stack: Current call stack. Recursive calls will build this up.
        """
        if stack is None:
            stack = []

        if isinstance(span, LiveSpanCall):
            if span.live_func is None:
                raise ValueError(f"Span {span} has no function.")

            path = get_method_path(obj=span.live_obj, func=span.live_func)

            frame_ident = record_schema.RecordAppCallMethod(
                path=path
                if path is not None
                else serial_utils.Lens().static,  # placeholder path for functions
                method=pyschema_utils.Method.of_method(
                    span.live_func, obj=span.live_obj, cls=span.live_cls
                ),
            )

            stack = stack + [frame_ident]
            span_stacks[span] = stack

        for subspan in span.iter_children(transitive=False):
            Tracer._fill_stacks(
                subspan,
                stack=stack,
                get_method_path=get_method_path,
                span_stacks=span_stacks,
            )

    def _call_of_spancall(
        self, span: LiveSpanCall, stack: List[record_schema.RecordAppCallMethod]
    ) -> record_schema.RecordAppCall:
        """Convert a LiveSpanCall to a RecordAppCall."""

        args = (
            dict(span.live_bindings.arguments)
            if span.live_bindings is not None
            else {}
        )
        if "self" in args:
            del args["self"]  # remove self

        assert span.start_timestamp is not None
        if span.end_timestamp is None:
            logger.warning(
                "Span %s has no end timestamp. It might not have yet finished recording.",
                span,
            )

        return record_schema.RecordAppCall(
            call_id=str(span.call_id),
            stack=stack,
            args={k: json_utils.jsonify(v) for k, v in args.items()},
            rets=json_utils.jsonify(span.live_ret),
            error=str(span.live_error),
            perf=base_schema.Perf(
                start_time=span.start_timestamp,
                end_time=span.end_timestamp,
            ),
            pid=span.process_id,
            tid=span.thread_id,
        )

    def record_of_root_span(
        self, recording: Any, root_span: LiveRecordRoot
    ) -> Tuple[record_schema.Record]:
        """Convert a root span to a record.

        This span has to be a call span so we can extract things like main input and output.
        """

        assert isinstance(root_span, LiveRecordRoot), type(root_span)

        # avoiding circular imports
        from trulens.experimental.otel_tracing.core import sem as core_sem

        app = recording.app

        # Use the record_id created during tracing.
        record_id = root_span.trace_record_id

        span_stacks = {}

        self._fill_stacks(
            root_span,
            span_stacks=span_stacks,
            get_method_path=app.get_method_path,
        )

        root_perf = base_schema.Perf(
            start_time=root_span.start_timestamp,
            end_time=root_span.end_timestamp,
        )

        total_cost = root_span.total_cost()

        calls = []
        spans = [core_sem.TypedSpan.semanticize(root_span)]

        for span in root_span.iter_children(include_phantom=True):
            if isinstance(span, LiveSpanCall):
                calls.append(
                    self._call_of_spancall(span, stack=span_stacks[span])
                )

            spans.append(core_sem.TypedSpan.semanticize(span))

        root_call_span = root_span.children_spans[
            0
        ]  # there should be exactly one

        bindings = root_call_span.live_bindings
        main_error = root_call_span.live_error

        if bindings is not None:
            main_input = app.main_input(
                func=root_call_span.live_func,
                sig=root_call_span.live_sig,
                bindings=root_call_span.live_bindings,
            )
            if main_error is None:
                main_output = app.main_output(
                    func=root_call_span.live_func,
                    sig=root_call_span.live_sig,
                    bindings=root_call_span.live_bindings,
                    ret=root_call_span.live_ret,
                )
            else:
                main_output = None
        else:
            main_input = None
            main_output = None

        record = record_schema.Record(
            record_id=record_id,
            app_id=app.app_id,
            main_input=json_utils.jsonify(main_input),
            main_output=json_utils.jsonify(main_output),
            main_error=json_utils.jsonify(main_error),
            calls=calls,
            perf=root_perf,
            cost=total_cost,
            experimental_otel_spans=spans,
        )

        return record

    @staticmethod
    def find_each_child(span: Span, span_filter: Callable) -> Iterable[Span]:
        """For each family rooted at each child of this span, find the top-most
        span that satisfies the filter."""

        for child_span in span.children_spans:
            if span_filter(child_span):
                yield child_span
            else:
                yield from Tracer.find_each_child(child_span, span_filter)

    def records_of_recording(
        self, recording: PhantomSpanRecordingContext
    ) -> Iterable[record_schema.Record]:
        """Convert a recording based on spans to a list of records."""

        for root_span in Tracer.find_each_child(
            span=recording, span_filter=lambda s: isinstance(s, LiveRecordRoot)
        ):
            assert isinstance(root_span, LiveRecordRoot), type(root_span)
            yield self.record_of_root_span(
                recording=recording, root_span=root_span
            )

    @contextlib.contextmanager
    def _span(self, cls: Type[S], **kwargs) -> ContextManager[S]:
        with self.start_span(cls=cls, **kwargs) as span:
            with python_utils.with_context({
                self._span_context_cvar: span.context
            }):
                yield span

    @contextlib.asynccontextmanager
    async def _aspan(self, cls: Type[S], **kwargs) -> ContextManager[S]:
        async with self.start_span(cls=cls, **kwargs) as span:
            async with python_utils.awith_context({
                self._span_context_cvar: span.context
            }):
                yield span

    # context manager
    def recording(self) -> ContextManager[PhantomSpanRecordingContext]:
        return self._span(
            name="trulens.recording", cls=PhantomSpanRecordingContext
        )

    # context manager
    def method(self, method_name: str) -> ContextManager[LiveSpanCall]:
        return self._span(name="trulens.call." + method_name, cls=LiveSpanCall)

    # context manager
    def cost(
        self, method_name: str, cost: Optional[base_schema.Cost] = None
    ) -> ContextManager[LiveSpanCallWithCost]:
        return self._span(
            name="trulens.call." + method_name,
            cls=LiveSpanCallWithCost,
            cost=cost,
        )

    # context manager
    def phantom(self) -> ContextManager[PhantomSpan]:
        return self._span(name="trulens.phantom", cls=PhantomSpan)

    # context manager
    async def arecording(self) -> ContextManager[PhantomSpanRecordingContext]:
        return self._aspan(
            name="trulens.recording", cls=PhantomSpanRecordingContext
        )

    # context manager
    async def amethod(self, method_name: str) -> ContextManager[LiveSpanCall]:
        return self._aspan(name="trulens.call." + method_name, cls=LiveSpanCall)

    # context manager
    async def acost(
        self, method_name: str, cost: Optional[base_schema.Cost] = None
    ) -> ContextManager[LiveSpanCallWithCost]:
        return self._aspan(
            name="trulens.call." + method_name,
            cls=LiveSpanCallWithCost,
            cost=cost,
        )

    # context manager
    async def aphantom(self) -> ContextManager[PhantomSpan]:
        return self._aspan(name="trulens.phantom", cls=PhantomSpan)


class TracerProvider(
    core_otel.TracerProvider, metaclass=python_utils.PydanticSingletonMeta
):
    """TruLens additions on top of [OTEL TracerProvider][opentelemetry.trace.TracerProvider]."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _trace_id: types_schema.TraceID.PY_TYPE = pydantic.PrivateAttr(
        default_factory=types_schema.TraceID.default_py
    )

    def __str__(self):
        # Pydantic will not print anything useful otherwise.
        return f"{self.__module__}.{type(self).__name__}()"

    @property
    def trace_id(self) -> types_schema.TraceID.PY_TYPE:
        return self._trace_id

    # Overrides core_otel.TracerProvider._tracer_class
    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(default=Tracer)

    _tracers: Dict[str, Tracer] = pydantic.PrivateAttr(default_factory=dict)

    _spans: Dict[SpanContext, Span] = pydantic.PrivateAttr(default_factory=dict)

    @property
    def spans(self) -> Dict[SpanContext, Span]:
        return self._spans

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[types_api.Attributes] = None,
    ):
        if instrumenting_module_name in self._tracers:
            return self._tracers[instrumenting_module_name]

        tracer = super().get_tracer(
            instrumenting_module_name=instrumenting_module_name,
            instrumenting_library_version=instrumenting_library_version,
            attributes=attributes,
            schema_url=schema_url,
        )

        self._tracers[instrumenting_module_name] = tracer

        return tracer


tracer_provider = TracerProvider()
"""Global tracer provider.
All trulens tracers are made by this provider even if a different one is
configured for OTEL.
"""


@functools.cache
def trulens_tracer():
    from trulens.core import __version__

    return tracer_provider.get_tracer(
        instrumenting_module_name="trulens.experimental.otel_tracing.core.trace",
        instrumenting_library_version=__version__,
    )


class TracingCallbacks(wrap_utils.CallableCallbacks[R], Generic[R, S]):
    """Extension of CallableCallbacks that adds tracing to the wrapped callable
    as implemented using tracer and spans."""

    def __init__(
        self,
        func_name: str,
        span_type: Type[S] = LiveSpanCall,
        enter_contexts: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """
        Args:
            enter_contexts: Whether to enter the context managers in this class
                init. If a subclass needs to add more context managers before
                entering, set this flag to false in `super().__init__` and then
                call `self._enter_contexts()` in own subclass `__init__`.
        """

        super().__init__(**kwargs)

        self.func_name: str = func_name

        self.obj: Optional[object] = None
        self.obj_cls: Optional[Type] = None
        self.obj_id: Optional[int] = None

        if not issubclass(span_type, LiveSpanCall):
            raise ValueError("span_type must be a subclass of LiveSpanCall.")

        self.span_context: ContextManager[LiveSpanCall] = (
            trulens_tracer()._span(span_type, name="trulens.call." + func_name)
        )
        # Will be filled in by _enter_contexts.
        self.span: Optional[LiveSpanCall] = None

        # Keeping track of possibly multiple contexts for subclasses to add
        # more.
        self.context_managers: List[ContextManager[LiveSpanCall]] = [
            self.span_context
        ]
        self.spans: List[Span] = []  # keep track of the spans we enter

        if enter_contexts:
            self._enter_contexts()

    def _enter_contexts(self):
        """Enter all of the context managers registered in this class.

        This includes the span for this callback but might include others if
        subclassed.
        """

        for context_manager in self.context_managers:
            self.spans.append(context_manager.__enter__())

        # Last context we enter is the span we track in this class. Subclasses
        # might have inserted earlier spans.
        self.span = self.spans[-1]

        # Propagate some fields from parent. Note that these may be updated by
        # the subclass of this callback class when new record roots get added.
        parent_span = self.span.parent_span
        if parent_span is not None:
            if isinstance(parent_span, Span):
                self.span.trace_record_ids = parent_span.trace_record_ids
            if isinstance(parent_span, LiveSpan):
                self.span.live_apps = parent_span.live_apps

    def _exit_contexts(self, error: Optional[Exception]) -> Optional[Exception]:
        """Exit all of the context managers registered in this class given the
        innermost context's exception optionally.

        Returns the unhandled error if the managers did not absorb it.
        """

        # Exit the contexts starting from the innermost one.
        for context_manager in self.context_managers[::-1]:
            if error is not None:
                try:
                    if context_manager.__exit__(
                        type(error), error, error.__traceback__
                    ):
                        # If the context absorbed the error, we don't propagate the
                        # error to outer contexts.
                        error = None

                except Exception as next_error:
                    # Manager might have absorbed the error but raised another
                    # so this error may not be the same as the original. While
                    # python docs say not to do this, it may happen due to bad
                    # exit implementation or just people not following the spec.
                    error = next_error

            else:
                self.span_context.__exit__(None, None, None)

        return error

    def on_callable_call(
        self, bindings: inspect.BoundArguments, **kwargs: Dict[str, Any]
    ) -> inspect.BoundArguments:
        temp = super().on_callable_call(bindings=bindings, **kwargs)

        if "self" in bindings.arguments:
            # TODO: need some generalization
            self.obj = bindings.arguments["self"]
            self.obj_cls = type(self.obj)
            self.obj_id = id(self.obj)
        else:
            logger.warning("No self in bindings for %s.", self)

        span = self.span
        assert span is not None, "Contexts not yet entered."
        span.process_id = os.getpid()
        span.thread_id = th.get_native_id()

        return temp

    def on_callable_end(self):
        super().on_callable_end()

        span = self.span

        # LiveSpanCall attributes
        span.call_id = self.call_id
        span.live_obj = self.obj
        span.live_cls = self.obj_cls
        span.live_func = self.func
        span.live_args = self.call_args
        span.live_kwargs = self.call_kwargs
        span.live_bindings = self.bindings
        span.live_sig = self.sig
        span.live_ret = self.ret
        span.live_error = self._exit_contexts(self.error)


class _RecordingContext:
    """Manager of the creation of records from record calls.

    An instance of this class is produced when using an
    [App][trulens_eval.app.App] as a context mananger, i.e.:
    Example:
        ```python
        app = ...  # your app
        truapp: TruChain = TruChain(app, ...) # recorder for LangChain apps
        with truapp as recorder:
            app.invoke(...) # use your app
        recorder: RecordingContext
        ```

    Each instance of this class produces a record for every "root" instrumented
    method called. Root method here means the first instrumented method in a
    call stack. Note that there may be more than one of these contexts in play
    at the same time due to:
    - More than one wrapper of the same app.
    - More than one context manager ("with" statement) surrounding calls to the
      same app.
    - Calls to "with_record" on methods that themselves contain recording.
    - Calls to apps that use trulens internally to track records in any of the
      supported ways.
    - Combinations of the above.
    """

    def __init__(
        self,
        app: _WithInstrumentCallbacks,
        record_metadata: serial_utils.JSON = None,
        tracer: Optional[Tracer] = None,
        span: Optional[PhantomSpanRecordingContext] = None,
        span_ctx: Optional[SpanContext] = None,
    ):
        self.calls: Dict[types_schema.CallID, record_schema.RecordAppCall] = {}
        """A record (in terms of its RecordAppCall) in process of being created.

        Storing as a map as we want to override calls with the same id which may
        happen due to methods producing awaitables or generators. These result
        in calls before the awaitables are awaited and then get updated after
        the result is ready.
        """
        # TODEP: To deprecated after migration to span-based tracing.

        self.records: List[record_schema.Record] = []
        """Completed records."""

        self.lock: Lock = Lock()
        """Lock blocking access to `records` when adding calls or
        finishing a record."""

        self.token: Optional[contextvars.Token] = None
        """Token for context management."""

        self.app: _WithInstrumentCallbacks = app
        """App for which we are recording."""

        self.record_metadata = record_metadata
        """Metadata to attach to all records produced in this context."""

        self.tracer: Optional[Tracer] = tracer
        """EXPERIMENTAL(otel_tracing): OTEL-like tracer for recording.
        """

        self.span: Optional[PhantomSpanRecordingContext] = span
        """EXPERIMENTAL(otel_tracing): Span that represents a recording context
        (the with block)."""

        self.span_ctx = span_ctx
        """EXPERIMENTAL(otel_tracing): The context manager for the above span.
        """

    @property
    def spans(self) -> Dict[SpanContext, Span]:
        """EXPERIMENTAL(otel_tracing): Get the spans of the tracer in this context."""

        if self.tracer is None:
            return {}

        return self.tracer.spans

    def __iter__(self):
        return iter(self.records)

    def get(self) -> record_schema.Record:
        """Get the single record only if there was exactly one or throw
        an error otherwise."""

        if len(self.records) == 0:
            raise RuntimeError("Recording context did not record any records.")

        if len(self.records) > 1:
            raise RuntimeError(
                "Recording context recorded more than 1 record. "
                "You can get them with ctx.records, ctx[i], or `for r in ctx: ...`."
            )

        return self.records[0]

    def __getitem__(self, idx: int) -> record_schema.Record:
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def __hash__(self) -> int:
        # The same app can have multiple recording contexts.
        return hash(id(self.app)) + hash(id(self.records))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def add_call(self, call: record_schema.RecordAppCall):
        """Add the given call to the currently tracked call list."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            # NOTE: This might override existing call record which happens when
            # processing calls with awaitable or generator results.
            self.calls[call.call_id] = call

    def finish_record(
        self,
        calls_to_record: Callable[
            [
                List[record_schema.RecordAppCall],
                types_schema.Metadata,
                Optional[record_schema.Record],
            ],
            record_schema.Record,
        ],
        existing_record: Optional[record_schema.Record] = None,
    ):
        """Run the given function to build a record from the tracked calls and any
        pre-specified metadata."""
        # TODEP: To deprecated after migration to span-based tracing.

        with self.lock:
            record = calls_to_record(
                list(self.calls.values()), self.record_metadata, existing_record
            )
            self.calls = {}

            if existing_record is None:
                # If existing record was given, we assume it was already
                # inserted into this list.
                self.records.append(record)

        return record


class _WithInstrumentCallbacks:
    """Abstract definition of callbacks invoked by Instrument during
    instrumentation or when instrumented methods are called.

    Needs to be mixed into [App][trulens_eval.app.App].
    """

    # Called during instrumentation.
    def on_method_instrumented(
        self, obj: object, func: Callable, path: serial_utils.Lens
    ):
        """Callback to be called by instrumentation system for every function
        requested to be instrumented.

        Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).

            path: The path of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def get_method_path(self, obj: object, func: Callable) -> serial_utils.Lens:
        """Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, serial_utils.Lens]]:
        """EXPERIMENTAL(otel_tracing): Get the methods (rather the inner
        functions) matching the given `func` and the path of each.

        Args:
            func: The function to match.
        """

        raise NotImplementedError

    # Called after recording of an invocation.
    def _on_new_root_span(
        self,
        ctx: _RecordingContext,
        root_span: LiveSpanCall,
    ) -> record_schema.Record:
        """EXPERIMENTAL(otel_tracing): Called by instrumented methods if they
        are root calls (first instrumented methods in a call stack).

        Args:
            ctx: The context of the recording.

            root_span: The root span that was recorded.
        """
        # EXPERIMENTAL(otel_tracing)

        raise NotImplementedError


class AppTracingCallbacks(TracingCallbacks[R, S]):
    """Extension to TracingCallbacks that keep track of apps that are
    instrumenting their constituent calls.

    Also inserts LiveRecordRoot spans
    """

    @classmethod
    def on_callable_wrapped(
        cls,
        wrapper: Callable[..., R],
        app: _WithInstrumentCallbacks,
        **kwargs: Dict[str, Any],
    ):
        # Adds the requesting app to the list of apps the wrapper is
        # instrumented for.

        if not python_utils.safe_hasattr(wrapper, APPS):
            apps: weakref.WeakSet[_WithInstrumentCallbacks] = weakref.WeakSet()
            setattr(wrapper, APPS, apps)
        else:
            apps = python_utils.safe_getattr(wrapper, APPS)

        apps.add(app)

        return super().on_callable_wrapped(wrapper=wrapper, **kwargs)

    def __init__(
        self,
        span_type: Type[Span] = LiveSpanCall,
        **kwargs: Dict[str, Any],
    ):
        # Do not enter the context managers in the superclass init as we need to
        # add another outer one possibly depending on the below logic.
        super().__init__(span_type=span_type, enter_contexts=False, **kwargs)

        # Get all of the apps that have instrumented this call.
        apps = python_utils.safe_getattr(self.wrapper, APPS)

        self.trace_root_span_context_managers: List[ContextManager] = []

        # Special handling of app calls: if they are the first app call in the
        # stack, they are wrapped in a trace root span. This is per-app so
        # mutliple new spans might be inserted.

        current_span = trulens_tracer().current_span

        record_map = {}
        started_apps: weakref.WeakSet[Any] = weakref.WeakSet()  # Any = App

        if current_span is None:
            pass
        else:
            if isinstance(current_span, Span):
                record_map.update(current_span.trace_record_ids)

            if isinstance(current_span, LiveSpan):
                started_apps = started_apps.union(current_span.live_apps)

            # Check which apps already have a root span in the ancestors:
            """
            for ancestor in current_span.iter_ancestors():
                if isinstance(ancestor, LiveRecordRoot):
                    assert (
                        ancestor.live_app is not None
                    ), "Root span has no app."
                    live_app = ancestor.live_app()
                    if live_app is not None:
                        started_apps.add(live_app)  # was weakref
                        record_map[live_app.app_id] = ancestor.record_id
                    else:
                        logger.warning(
                            "App in span %s got collected before we got a chance to serialize the span.",
                            ancestor,
                        )
            """

        # Now for each app that is not yet in record_ids, create a span context
        # manager for it and add it to record_ids of the new created span.

        for app in set(apps).difference(started_apps):
            new_record_id = types_schema.TraceRecordID.default_py()
            record_map[app.app_id] = new_record_id
            self.trace_root_span_context_managers.append(
                trulens_tracer()._span(
                    cls=LiveRecordRoot,
                    name=truconv.SpanAttributes.RECORD_ROOT.SPAN_NAME,
                    live_app=weakref.ref(app),
                    trace_record_id=new_record_id,
                    trace_record_ids=record_map,
                )
            )
            started_apps.add(app)

            print(
                f"Adding root span for app {app.app_name} at call to {self.func_name}."
            )

        # Importantly, add the managers for the trace root `before` the span
        # managed by TracingCallbacks. This makes sure the root spans are the
        # parents of the call span. Unsure if the order between the root spans
        # matters much.
        self.context_managers = (
            self.trace_root_span_context_managers + self.context_managers
        )

        # Finally enter the contexts, possibly including the ones we just added.
        self._enter_contexts()

        assert self.span is not None, "Contexts not yet entered."

        # Make note of all the apps the main span is recording for and the app
        # to record map.
        if issubclass(span_type, Span):
            self.span.trace_record_ids = record_map

        if issubclass(span_type, LiveSpan):
            self.span.live_apps = started_apps
