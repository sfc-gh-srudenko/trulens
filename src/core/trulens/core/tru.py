from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from datetime import datetime
from datetime import timedelta
import json
import logging
from multiprocessing import Process
from pprint import PrettyPrinter
import re
import threading
from threading import Thread
from time import sleep
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import humanize
import pandas
from tqdm.auto import tqdm
from trulens.core import feedback
from trulens.core.database.base import DB
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import python
from trulens.core.utils import serial
from trulens.core.utils import threading as tru_threading
from trulens.core.utils.imports import REQUIREMENT_SNOWFLAKE
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.python import Future  # code style exception
from trulens.core.utils.python import OpaqueWrapper

with OptionalImports(messages=REQUIREMENT_SNOWFLAKE):
    from snowflake.core import CreateMode
    from snowflake.core import Root
    from snowflake.core.schema import Schema
    from snowflake.snowpark import Session
    from snowflake.sqlalchemy import URL

pp = PrettyPrinter()

logger = logging.getLogger(__name__)


def humanize_seconds(seconds: float):
    return humanize.naturaldelta(timedelta(seconds=seconds))


class Tru(python.SingletonPerName):
    """Tru is the main class that provides an entry point to TruLens.

    Tru lets you:

    - Log app prompts and outputs
    - Log app Metadata
    - Run and log feedback functions
    - Run streamlit dashboard to view experiment results

    By default, all data is logged to the current working directory to
    `"default.sqlite"`. Data can be logged to a SQLAlchemy-compatible url
    referred to by `database_url`.

    Supported App Types:
        [TruChain][trulens.instrument.langchain.TruChain]: Langchain
            apps.

        [TruLlama][trulens.instrument.llamaindex.TruLlama]: Llama Index
            apps.

        [TruRails][trulens.instrument.nemo.TruRails]: NeMo Guardrails apps.

        [TruBasicApp][trulens.core.TruBasicApp]:
            Basic apps defined solely using a function from `str` to `str`.

        [TruCustomApp][trulens.core.TruCustomApp]:
            Custom apps containing custom structures and methods. Requires annotation
            of methods to instrument.

        [TruVirtual][trulens.core.TruVirtual]: Virtual
            apps that do not have a real app to instrument but have a virtual
            structure and can log existing captured data as if they were trulens
            records.

    Args:
        app_name (str, optional): The name of the app. Defaults to "default".

        database (Optional[DB], optional): Database to use. If not provided, an
            [SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] database
            will be initialized based on the other arguments. Defaults to None.

        database_url (Optional[str], optional): Database URL. Defaults to a local SQLite
            database file at `"default.sqlite"` See [this
            article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)
            on SQLAlchemy database URLs. (defaults to
            `sqlite://{app_name}.sqlite`).

        database_redact_keys (bool, optional): Whether to redact secret keys in data to be
            written to database (defaults to `False`)

        database_prefix (Optional[str], optional): Prefix for table names for trulens to use.
            May be useful in some databases hosting other apps. Defaults to None.

        database_args (Optional[Dict[str, Any]], optional): Additional arguments to pass to the database constructor. Defaults to None.

        database_check_revision (bool, optional): Whether to check the database revision on
            init. This prompt determine whether database migration is required. Defaults to True.

        snowflake_connection_parameters (Optional[Dict[str, str]], optional): Connection arguments to Snowflake database to use. Defaults to None.
    """

    RETRY_RUNNING_SECONDS: float = 60.0
    """How long to wait (in seconds) before restarting a feedback function that has already started

    A feedback function execution that has started may have stalled or failed in a bad way that did not record the
    failure.

    See also:
        [start_evaluator][trulens.core.tru.Tru.start_evaluator]

        [DEFERRED][trulens.core.schema.feedback.FeedbackMode.DEFERRED]
    """

    RETRY_FAILED_SECONDS: float = 5 * 60.0
    """How long to wait (in seconds) to retry a failed feedback function run."""

    DEFERRED_NUM_RUNS: int = 32
    """Number of futures to wait for when evaluating deferred feedback functions."""

    db: Union[DB, OpaqueWrapper[DB]]
    """Database supporting this workspace.

    Will be an opaque wrapper if it is not ready to use due to migration requirements.
    """

    _dashboard_urls: Optional[str] = None

    _evaluator_proc: Optional[Union[Process, Thread]] = None
    """[Process][multiprocessing.Process] or [Thread][threading.Thread] of the deferred feedback evaluator if started.

        Is set to `None` if evaluator is not running.
    """

    _dashboard_proc: Optional[Process] = None
    """[Process][multiprocessing.Process] executing the dashboard streamlit app.

    Is set to `None` if not executing.
    """

    _evaluator_stop: Optional[threading.Event] = None
    """Event for stopping the deferred evaluator which runs in another thread."""

    def __new__(cls, *args, **kwargs) -> Tru:
        inst = super().__new__(cls, *args, **kwargs)
        assert isinstance(inst, Tru)
        return inst

    def __init__(
        self,
        app_name: str = "default",
        database: Optional[DB] = None,
        database_url: Optional[str] = None,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
        snowflake_connection_parameters: Optional[Dict[str, str]] = None,
    ):
        database_args = database_args or {}

        if python.safe_hasattr(self, "db"):
            # Already initialized by SingletonByName mechanism. Give warning if
            # any option was specified (not None) as it will be ignored.
            for v in database_args.values():
                if v is not None:
                    logger.warning(
                        "Tru was already initialized. Cannot change database configuration after initialization."
                    )
                    self.warning()
                    break
            return

        if database is not None:
            if not isinstance(database, DB):
                raise ValueError(
                    "`database` must be a `trulens.core.database.base.DB` instance."
                )

            self.db = database
        else:
            self.db = self._init_db(
                app_name=app_name,
                database_args=database_args,
                snowflake_connection_parameters=snowflake_connection_parameters,
                database_url=database_url,
                database_redact_keys=database_redact_keys,
                database_prefix=database_prefix,
            )

        if database_check_revision:
            try:
                self.db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self.db = OpaqueWrapper(obj=self.db, e=e)

    def _init_db(
        self,
        app_name: str,
        database_args: Dict[str, Any],
        snowflake_connection_parameters: Optional[Dict[str, str]] = None,
        database_url: Optional[str] = None,
        database_redact_keys: bool = False,
        database_prefix: Optional[str] = None,
    ) -> DB:
        if snowflake_connection_parameters is not None:
            if database_url is not None:
                raise ValueError(
                    "`database_url` must be `None` if `snowflake_connection_parameters` is set!"
                )
            if not app_name:
                raise ValueError(
                    "`app_name` must be set if `snowflake_connection_parameters` is set!"
                )
            schema_name = self._validate_and_compute_schema_name(app_name)
            database_url = self._create_snowflake_database_url(
                snowflake_connection_parameters, schema_name
            )
        else:
            database_url = database_url or f"sqlite:///{app_name}.sqlite"

        database_args.update(
            {
                k: v
                for k, v in {
                    "database_url": database_url,
                    "database_redact_keys": database_redact_keys,
                    "database_prefix": database_prefix,
                }.items()
                if v is not None
            }
        )
        return SQLAlchemyDB.from_tru_args(**database_args)

    @staticmethod
    def _validate_and_compute_schema_name(app_name: str):
        if not re.match(r"^[A-Za-z0-9_]+$", app_name):
            raise ValueError(
                "`app_name` must contain only alphanumeric and underscore characters!"
            )
        return f"TRULENS_APP__{app_name.upper()}"

    @staticmethod
    def _create_snowflake_database_url(
        snowflake_connection_parameters: Dict[str, str], schema_name: str
    ) -> str:
        Tru._create_snowflake_schema_if_not_exists(
            snowflake_connection_parameters, schema_name
        )
        return URL(
            account=snowflake_connection_parameters["account"],
            user=snowflake_connection_parameters["user"],
            password=snowflake_connection_parameters["password"],
            database=snowflake_connection_parameters["database"],
            schema=schema_name,
            warehouse=snowflake_connection_parameters.get("warehouse", None),
            role=snowflake_connection_parameters.get("role", None),
        )

    @staticmethod
    def _create_snowflake_schema_if_not_exists(
        snowflake_connection_parameters: Dict[str, str],
        schema_name: str,
    ):
        session = Session.builder.configs(
            snowflake_connection_parameters
        ).create()
        root = Root(session)
        schema = Schema(name=schema_name)
        root.databases[
            snowflake_connection_parameters["database"]
        ].schemas.create(schema, mode=CreateMode.if_not_exists)

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """

        if isinstance(self.db, OpaqueWrapper):
            db = self.db.unwrap()
        elif isinstance(self.db, DB):
            db = self.db
        else:
            raise RuntimeError("Unhandled database type.")

        db.reset_database()
        self.db = db

    def migrate_database(self, **kwargs: Dict[str, Any]):
        """Migrates the database.

        This should be run whenever there are breaking changes in a database
        created with an older version of _trulens_.

        Args:
            **kwargs: Keyword arguments to pass to
                [migrate_database][trulens.core.database.base.DB.migrate_database]
                of the current database.

        See [DB.migrate_database][trulens.core.database.base.DB.migrate_database].
        """

        if isinstance(self.db, OpaqueWrapper):
            db = self.db.unwrap()
        elif isinstance(self.db, DB):
            db = self.db
        else:
            raise RuntimeError("Unhandled database type.")

        db.migrate_database(**kwargs)
        self.db = db

    def add_record(
        self, record: Optional[mod_record_schema.Record] = None, **kwargs: dict
    ) -> mod_types_schema.RecordID:
        """Add a record to the database.

        Args:
            record: The record to add.

            **kwargs: [Record][trulens.core.schema.record.Record] fields to add to the
                given record or a new record if no `record` provided.

        Returns:
            Unique record identifier [str][] .

        """

        if record is None:
            record = mod_record_schema.Record(**kwargs)
        else:
            record.update(**kwargs)

        return self.db.insert_record(record=record)

    update_record = add_record

    # TODO: this method is used by app.py, which represents poor code
    # organization.
    def _submit_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app_version: Optional[mod_app_schema.AppVersionDefinition] = None,
        on_done: Optional[
            Callable[
                [
                    Union[
                        mod_feedback_schema.FeedbackResult,
                        Future[mod_feedback_schema.FeedbackResult],
                    ],
                ],
                None,
            ]
        ] = None,
    ) -> List[
        Tuple[feedback.Feedback, Future[mod_feedback_schema.FeedbackResult]]
    ]:
        """Schedules to run the given feedback functions.

        Args:
            record: The record on which to evaluate the feedback functions.

            feedback_functions: A collection of feedback functions to evaluate.

            app_version: The app version that produced the given record. If not provided, it is
                looked up from the database of this `Tru` instance

            on_done: A callback to call when each feedback function is done.

        Returns:

            List[Tuple[feedback.Feedback, Future[schema.FeedbackResult]]]

            Produces a list of tuples where the first item in each tuple is the
            feedback function and the second is the future of the feedback result.
        """

        version_tag = record.version_tag

        self.db: DB

        if app_version is None:
            app_version = mod_app_schema.AppVersionDefinition.model_validate(
                self.db.get_app_version(version_tag=version_tag)
            )
            if app_version is None:
                raise RuntimeError(
                    f"Version `{version_tag}` not present in db. "
                    "Either add it with `tru.add_app` or provide `app_json` to `tru.run_feedback_functions`."
                )

        else:
            assert (
                version_tag == app_version.version_tag
            ), "Record was produced by a different app."

            if (
                self.db.get_app_version(version_tag=app_version.version_tag)
                is None
            ):
                logger.warning(
                    f"Version `{version_tag}` was not present in database. Adding it."
                )
                self.add_version(app_version=app_version)

        feedbacks_and_futures = []

        tp: tru_threading.TP = tru_threading.TP()

        for ffunc in feedback_functions:
            # Run feedback function and the on_done callback. This makes sure
            # that Future.result() returns only after on_done has finished.
            def run_and_call_callback(
                ffunc: feedback.Feedback,
                app_version: mod_app_schema.AppVersionDefinition,
                record: mod_record_schema.Record,
            ):
                temp = ffunc.run(app_version=app_version, record=record)
                if on_done is not None:
                    try:
                        on_done(temp)
                    finally:
                        return temp

                return temp

            fut: Future[mod_feedback_schema.FeedbackResult] = tp.submit(
                run_and_call_callback,
                ffunc=ffunc,
                app_version=app_version,
                record=record,
            )

            # Have to roll the on_done callback into the submitted function
            # because the result() is returned before callback runs otherwise.
            # We want to do db work before result is returned.

            # if on_done is not None:
            #    fut.add_done_callback(on_done)

            feedbacks_and_futures.append((ffunc, fut))

        return feedbacks_and_futures

    def run_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app_version: Optional[mod_app_schema.AppVersionDefinition] = None,
        wait: bool = True,
    ) -> Union[
        Iterable[mod_feedback_schema.FeedbackResult],
        Iterable[Future[mod_feedback_schema.FeedbackResult]],
    ]:
        """Run a collection of feedback functions and report their result.

        Args:
            record: The record on which to evaluate the feedback
                functions.

            app_version: The app version that produced the given record.
                If not provided, it is looked up from the given database `db`.

            feedback_functions: A collection of feedback
                functions to evaluate.

            wait: If set (default), will wait for results
                before returning.

        Yields:
            One result for each element of `feedback_functions` of
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] if `wait`
                is enabled (default) or [Future][concurrent.futures.Future] of
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] if `wait`
                is disabled.
        """

        if not isinstance(record, mod_record_schema.Record):
            raise ValueError(
                "`record` must be a `trulens.core.schema.record.Record` instance."
            )

        if not isinstance(feedback_functions, Sequence):
            raise ValueError("`feedback_functions` must be a sequence.")

        if not all(
            isinstance(ffunc, feedback.Feedback) for ffunc in feedback_functions
        ):
            raise ValueError(
                "`feedback_functions` must be a sequence of `trulens.core.Feedback` instances."
            )

        if not (
            app_version is None
            or isinstance(app_version, mod_app_schema.AppVersionDefinition)
        ):
            raise ValueError(
                "`app_version` must be a `trulens.core.schema.app.AppVersionDefinition` instance."
            )

        if not isinstance(wait, bool):
            raise ValueError("`wait` must be a bool.")

        future_feedback_map: Dict[
            Future[mod_feedback_schema.FeedbackResult], feedback.Feedback
        ] = {
            p[1]: p[0]
            for p in self._submit_feedback_functions(
                record=record,
                feedback_functions=feedback_functions,
                app_version=app_version,
            )
        }

        if wait:
            # In blocking mode, wait for futures to complete.
            for fut_result in futures.as_completed(future_feedback_map.keys()):
                # TODO: Do we want a version that gives the feedback for which
                # the result is being produced too? This is more useful in the
                # Future case as we cannot check associate a Future result to
                # its feedback before result is ready.

                # yield (future_feedback_map[fut_result], fut_result.result())
                yield fut_result.result()

        else:
            # In non-blocking, return the futures instead.
            for fut_result, _ in future_feedback_map.items():
                # TODO: see prior.

                # yield (feedback, fut_result)
                yield fut_result

    def add_version(
        self, app_version: mod_app_schema.AppVersionDefinition
    ) -> mod_types_schema.VersionTag:
        """
        Add an app to the database and return its unique id.

        Args:
            app: The app to add to the database.

        Returns:
            A unique app identifier [str][].

        """

        return self.db.insert_app_version(app_version=app_version)

    def delete_version(self, version_tag: mod_types_schema.VersionTag) -> None:
        """
        Deletes an version from the app based on its version tag.

        Args:
            version (schema.VersionTag): The unique identifier of the app version to be deleted.
        """
        self.db.delete_app_version(version_tag=version_tag)
        logger.info(f"App with ID {version_tag} has been successfully deleted.")

    def add_feedback(
        self,
        feedback_result_or_future: Optional[
            Union[
                mod_feedback_schema.FeedbackResult,
                Future[mod_feedback_schema.FeedbackResult],
            ]
        ] = None,
        **kwargs: dict,
    ) -> mod_types_schema.FeedbackResultID:
        """Add a single feedback result or future to the database and return its unique id.

        Args:
            feedback_result_or_future: If a [Future][concurrent.futures.Future]
                is given, call will wait for the result before adding it to the
                database. If `kwargs` are given and a
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] is also
                given, the `kwargs` will be used to update the
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] otherwise a
                new one will be created with `kwargs` as arguments to its
                constructor.

            **kwargs: Fields to add to the given feedback result or to create a
                new [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] with.

        Returns:
            A unique result identifier [str][].

        """

        if feedback_result_or_future is None:
            if "result" in kwargs and "status" not in kwargs:
                # If result already present, set status to done.
                kwargs["status"] = mod_feedback_schema.FeedbackResultStatus.DONE

            feedback_result_or_future = mod_feedback_schema.FeedbackResult(
                **kwargs
            )

        else:
            if isinstance(feedback_result_or_future, Future):
                futures.wait([feedback_result_or_future])
                feedback_result_or_future: mod_feedback_schema.FeedbackResult = feedback_result_or_future.result()

            elif isinstance(
                feedback_result_or_future, mod_feedback_schema.FeedbackResult
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown type {type(feedback_result_or_future)} in feedback_results."
                )

            feedback_result_or_future.update(**kwargs)

        return self.db.insert_feedback(
            feedback_result=feedback_result_or_future
        )

    def add_feedbacks(
        self,
        feedback_results: Iterable[
            Union[
                mod_feedback_schema.FeedbackResult,
                Future[mod_feedback_schema.FeedbackResult],
            ]
        ],
    ) -> List[mod_types_schema.FeedbackResultID]:
        """Add multiple feedback results to the database and return their unique ids.

        Args:
            feedback_results: An iterable with each iteration being a [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] or
                [Future][concurrent.futures.Future] of the same. Each given future will be waited.

        Returns:
            List of unique result identifiers [str][] in the same order as input
                `feedback_results`.
        """

        ids = []

        for feedback_result_or_future in feedback_results:
            ids.append(
                self.add_feedback(
                    feedback_result_or_future=feedback_result_or_future
                )
            )

        return ids

    def get_version(
        self, version_tag: mod_types_schema.VersionTag
    ) -> Optional[serial.JSONized[mod_app_schema.AppVersionDefinition]]:
        """Look up an app from the database.

        This method produces the JSON-ized version of the app. It can be deserialized back into an [AppVersionDefinition][trulens.core.schema.app.AppVersionDefinition] with [model_validate][pydantic.BaseModel.model_validate]:

        Example:
            ```python
            from trulens.core.schema import app
            app_json = tru.get_version(version="Custom Application v1")
            app = app.AppVersionDefinition.model_validate(app_json)
            ```

        Warning:
            Do not rely on deserializing into [App][trulens.core.app.App] as
            its implementations feature attributes not meant to be deserialized.

        Args:
            version_tag: The unique identifier [str] of the app to look up.

        Returns:
            JSON-ized version of the app.
        """

        return self.db.get_app_version(version_tag=version_tag)

    def get_all_versions(
        self,
    ) -> Iterable[serial.JSONized[mod_app_schema.AppVersionDefinition]]:
        """Look up all apps from the database.

        Returns:
            A list of JSON-ized version of all apps in the database.

        Warning:
            Same Deserialization caveats as [get_app][trulens.core.tru.Tru.get_app].
        """

        return self.db.get_app_versions()

    def get_records_and_feedback(
        self,
        versions: Optional[List[mod_types_schema.VersionTag]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pandas.DataFrame, List[str]]:
        """Get records, their feedback results, and feedback names.

        Args:
            versions: A list of version tags to filter records by. If empty or not given, records
            from all versions will be returned.

            offset: Record row offset.

            limit: Limit on the number of records to return.

        Returns:
            DataFrame of records with their feedback results.

            List of feedback names that are columns in the DataFrame.
        """

        if versions is None:
            versions = []

        df, feedback_columns = self.db.get_records_and_feedback(
            versions, offset=offset, limit=limit
        )

        return df, feedback_columns

    def get_leaderboard(
        self,
        versions: Optional[List[mod_types_schema.VersionTag]] = None,
        group_by_metadata_key: Optional[str] = None,
    ) -> pandas.DataFrame:
        """Get a leaderboard for the given apps.

        Args:
            versions: A list of app ids to filter records by. If empty or not given, all
                apps will be included in leaderboard.
            group_by_metadata_key: A key included in record metadata that you want to group results by.

        Returns:
            DataFrame of apps with their feedback results aggregated.
            If group_by_metadata_key is provided, the DataFrame will be grouped by the specified key.
        """

        if versions is None:
            versions = []

        df, feedback_cols = self.db.get_records_and_feedback(versions)

        col_agg_list = list(feedback_cols) + ["latency", "total_cost"]

        if group_by_metadata_key is not None:
            df["meta"] = [
                json.loads(df["record_json"][i])["meta"] for i in range(len(df))
            ]

            df[str(group_by_metadata_key)] = [
                item.get(group_by_metadata_key, None)
                if isinstance(item, dict)
                else None
                for item in df["meta"]
            ]
            return (
                df.groupby(["version_tag", str(group_by_metadata_key)])[
                    col_agg_list
                ]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )
        else:
            return (
                df.groupby("version_tag")[col_agg_list]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )

    def start_evaluator(
        self,
        restart: bool = False,
        fork: bool = False,
        disable_tqdm: bool = False,
    ) -> Union[Process, Thread]:
        """
        Start a deferred feedback function evaluation thread or process.

        Args:
            restart: If set, will stop the existing evaluator before starting a
                new one.

            fork: If set, will start the evaluator in a new process instead of a
                thread. NOT CURRENTLY SUPPORTED.

            disable_tqdm: If set, will disable progress bar logging from the evaluator.

        Returns:
            The started process or thread that is executing the deferred feedback
                evaluator.

        Relevant constants:
            [RETRY_RUNNING_SECONDS][trulens.core.tru.Tru.RETRY_RUNNING_SECONDS]

            [RETRY_FAILED_SECONDS][trulens.core.tru.Tru.RETRY_FAILED_SECONDS]

            [DEFERRED_NUM_RUNS][trulens.core.tru.Tru.DEFERRED_NUM_RUNS]

            [MAX_THREADS][trulens.core.utils.threading.TP.MAX_THREADS]
        """

        assert not fork, "Fork mode not yet implemented."

        if self._evaluator_proc is not None:
            if restart:
                self.stop_evaluator()
            else:
                raise RuntimeError(
                    "Evaluator is already running in this process."
                )

        if not fork:
            self._evaluator_stop = threading.Event()

        def runloop():
            assert self._evaluator_stop is not None

            print(
                f"Will keep max of "
                f"{self.DEFERRED_NUM_RUNS} feedback(s) running."
            )
            print(
                f"Tasks are spread among max of "
                f"{tru_threading.TP.MAX_THREADS} thread(s)."
            )
            print(
                f"Will rerun running feedbacks after "
                f"{humanize_seconds(self.RETRY_RUNNING_SECONDS)}."
            )
            print(
                f"Will rerun failed feedbacks after "
                f"{humanize_seconds(self.RETRY_FAILED_SECONDS)}."
            )

            total = 0

            # Getting total counts from the database to start off the tqdm
            # progress bar initial values so that they offer accurate
            # predictions initially after restarting the process.
            queue_stats = self.db.get_feedback_count_by_status()
            queue_done = (
                queue_stats.get(mod_feedback_schema.FeedbackResultStatus.DONE)
                or 0
            )
            queue_total = sum(queue_stats.values())

            # Show the overall counts from the database, not just what has been
            # looked at so far.
            tqdm_status = tqdm(
                desc="Feedback Status",
                initial=queue_done,
                unit="feedbacks",
                total=queue_total,
                postfix={
                    status.name: count for status, count in queue_stats.items()
                },
                disable=disable_tqdm,
            )

            # Show the status of the results so far.
            tqdm_total = tqdm(
                desc="Done Runs", initial=0, unit="runs", disable=disable_tqdm
            )

            # Show what is being waited for right now.
            tqdm_waiting = tqdm(
                desc="Waiting for Runs",
                initial=0,
                unit="runs",
                disable=disable_tqdm,
            )

            runs_stats = defaultdict(int)

            futures_map: Dict[
                Future[mod_feedback_schema.FeedbackResult], pandas.Series
            ] = dict()

            while fork or not self._evaluator_stop.is_set():
                if len(futures_map) < self.DEFERRED_NUM_RUNS:
                    # Get some new evals to run if some already completed by now.
                    new_futures: List[
                        Tuple[
                            pandas.Series,
                            Future[mod_feedback_schema.FeedbackResult],
                        ]
                    ] = feedback.Feedback.evaluate_deferred(
                        tru=self,
                        limit=self.DEFERRED_NUM_RUNS - len(futures_map),
                        shuffle=True,
                    )

                    # Will likely get some of the same ones that already have running.
                    for row, fut in new_futures:
                        if fut in futures_map:
                            # If the future is already in our set, check whether
                            # its status has changed and if so, note it in the
                            # runs_stats.
                            if futures_map[fut].status != row.status:
                                runs_stats[row.status.name] += 1

                        futures_map[fut] = row
                        total += 1

                    tqdm_total.total = total
                    tqdm_total.refresh()

                tqdm_waiting.total = self.DEFERRED_NUM_RUNS
                tqdm_waiting.n = len(futures_map)
                tqdm_waiting.refresh()

                # Note whether we have waited for some futures in this
                # iteration. Will control some extra wait time if there is no
                # work.
                did_wait = False

                if len(futures_map) > 0:
                    did_wait = True

                    futures_copy = list(futures_map.keys())

                    try:
                        for fut in futures.as_completed(
                            futures_copy, timeout=10
                        ):
                            del futures_map[fut]

                            tqdm_waiting.update(-1)
                            tqdm_total.update(1)

                            feedback_result = fut.result()
                            runs_stats[feedback_result.status.name] += 1

                    except futures.TimeoutError:
                        pass

                tqdm_total.set_postfix(
                    {name: count for name, count in runs_stats.items()}
                )

                queue_stats = self.db.get_feedback_count_by_status()
                queue_done = (
                    queue_stats.get(
                        mod_feedback_schema.FeedbackResultStatus.DONE
                    )
                    or 0
                )
                queue_total = sum(queue_stats.values())

                tqdm_status.n = queue_done
                tqdm_status.total = queue_total
                tqdm_status.set_postfix(
                    {
                        status.name: count
                        for status, count in queue_stats.items()
                    }
                )

                # Check if any of the running futures should be stopped.
                futures_copy = list(futures_map.keys())
                for fut in futures_copy:
                    row = futures_map[fut]

                    if fut.running():
                        # Not checking status here as this will be not yet be set
                        # correctly. The computation in the future updates the
                        # database but this object is outdated.

                        elapsed = datetime.now().timestamp() - row.last_ts
                        if elapsed > self.RETRY_RUNNING_SECONDS:
                            fut.cancel()

                            # Not an actual status, but would be nice to
                            # indicate cancellations in run stats:
                            runs_stats["CANCELLED"] += 1

                            del futures_map[fut]

                if not did_wait:
                    # Nothing to run/is running, wait a bit.
                    if fork:
                        sleep(10)
                    else:
                        self._evaluator_stop.wait(10)

            print("Evaluator stopped.")

        if fork:
            proc = Process(target=runloop)
        else:
            proc = Thread(target=runloop)
            proc.daemon = True

        # Start a persistent thread or process that evaluates feedback functions.

        self._evaluator_proc = proc
        proc.start()

        return proc

    run_evaluator = start_evaluator

    def stop_evaluator(self):
        """
        Stop the deferred feedback evaluation thread.
        """

        if self._evaluator_proc is None:
            raise RuntimeError("Evaluator not running this process.")

        if isinstance(self._evaluator_proc, Process):
            self._evaluator_proc.terminate()

        elif isinstance(self._evaluator_proc, Thread):
            self._evaluator_stop.set()
            self._evaluator_proc.join()
            self._evaluator_stop = None

        self._evaluator_proc = None
