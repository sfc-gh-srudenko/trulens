import abc
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from merkle_json import MerkleJson
import pandas as pd
from trulens.core.schema import types as mod_types_schema
from trulens.core.schema.app import AppDefinition
from trulens.core.schema.feedback import FeedbackDefinition
from trulens.core.schema.feedback import FeedbackResult
from trulens.core.schema.feedback import FeedbackResultStatus
from trulens.core.schema.record import Record
from trulens.core.utils.json import json_str_of_obj
from trulens.core.utils.serial import JSON
from trulens.core.utils.serial import JSONized
from trulens.core.utils.serial import SerialModel

mj = MerkleJson()
NoneType = type(None)

logger = logging.getLogger(__name__)

MULTI_CALL_NAME_DELIMITER = ":::"

DEFAULT_DATABASE_PREFIX: str = "trulens_"
"""Default prefix for table names for trulens to use.

This includes alembic's version table.
"""

DEFAULT_DATABASE_FILE: str = "default.sqlite"
"""Filename for default sqlite database.

The sqlalchemy url for this default local sqlite database is `sqlite:///default.sqlite`.
"""

DEFAULT_DATABASE_REDACT_KEYS: bool = False
"""Default value for option to redact secrets before writing out data to database."""


class DB(SerialModel, abc.ABC):
    """Abstract definition of databases used by trulens.

    [SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] is the main
    and default implementation of this interface.
    """

    redact_keys: bool = DEFAULT_DATABASE_REDACT_KEYS
    """Redact secrets before writing out data."""

    table_prefix: str = DEFAULT_DATABASE_PREFIX
    """Prefix for table names for trulens to use.

    May be useful in some databases where trulens is not the only app.
    """

    def _json_str_of_obj(self, obj: Any) -> str:
        return json_str_of_obj(obj, redact_keys=self.redact_keys)

    @abc.abstractmethod
    def reset_database(self):
        """Delete all data."""

        raise NotImplementedError()

    @abc.abstractmethod
    def migrate_database(self, prior_prefix: Optional[str] = None):
        """Migrade the stored data to the current configuration of the database.

        Args:
            prior_prefix: If given, the database is assumed to have been
                reconfigured from a database with the given prefix. If not
                given, it may be guessed if there is only one table in the
                database with the suffix `alembic_version`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def check_db_revision(self):
        """Check that the database is up to date with the current trulens
        version.

        Raises:
            ValueError: If the database is not up to date.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self,
        record: Record,
    ) -> mod_types_schema.RecordID:
        """
        Upsert a `record` into the database.

        Args:
            record: The record to insert or update.

        Returns:
            The id of the given record.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_app_version(
        self, app: AppDefinition
    ) -> mod_types_schema.AppVersion:
        """
        Upsert an `app` into the database.

        Args:
            app: The app to insert or update. Note that only the
                [AppDefinition][trulens.core.schema.app.AppDefinition] parts are serialized
                hence the type hint.

        Returns:
            The id of the given app.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_definition(
        self, feedback_definition: FeedbackDefinition
    ) -> mod_types_schema.FeedbackDefinitionID:
        """
        Upsert a `feedback_definition` into the database.

        Args:
            feedback_definition: The feedback definition to insert or update.
                Note that only the
                [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition]
                parts are serialized hence the type hint.

        Returns:
            The id of the given feedback definition.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
    ) -> pd.DataFrame:
        """Retrieve feedback definitions from the database.

        Args:
            feedback_definition_id: if provided, only the
                feedback definition with the given id is returned. Otherwise,
                all feedback definitions are returned.

        Returns:
            A dataframe with the feedback definitions.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        feedback_result: FeedbackResult,
    ) -> mod_types_schema.FeedbackResultID:
        """Upsert a `feedback_result` into the the database.

        Args:
            feedback_result: The feedback result to insert or update.

        Returns:
            The id of the given feedback result.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[FeedbackResultStatus, Sequence[FeedbackResultStatus]]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Get feedback results matching a set of optional criteria:

        Args:
            record_id: Get only the feedback for the given record id.

            feedback_result_id: Get only the feedback for the given feedback
                result id.

            feedback_definition_id: Get only the feedback for the given feedback
                definition id.

            status: Get only the feedback with the given status. If a sequence
                of statuses is given, all feedback with any of the given
                statuses are returned.

            last_ts_before: get only results with `last_ts` before the
                given datetime.

            offset: index of the first row to return.

            limit: limit the number of rows returned.

            shuffle: shuffle the rows before returning them.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_feedback_count_by_status(
        self,
        record_id: Optional[mod_types_schema.RecordID] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        feedback_definition_id: Optional[
            mod_types_schema.FeedbackDefinitionID
        ] = None,
        status: Optional[
            Union[FeedbackResultStatus, Sequence[FeedbackResultStatus]]
        ] = None,
        last_ts_before: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
    ) -> Dict[FeedbackResultStatus, int]:
        """Get count of feedback results matching a set of optional criteria grouped by
        their status.

        See [get_feedback][trulens.core.database.base.DB.get_feedback] for the meaning of
        the the arguments.

        Returns:
            A mapping of status to the count of feedback results of that status
                that match the given filters.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_app_version(
        self, app_version: mod_types_schema.AppVersion
    ) -> Optional[JSONized]:
        """Get the app version with the given version tag from the database.

        Returns:
            The JSON-ized version of the app with the given id. Deserialization
                can be done with
                [App.model_validate][trulens.core.app.App.model_validate].

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_app_versions(self) -> Iterable[JSON]:
        """Get all app versions."""

        raise NotImplementedError()

    @abc.abstractmethod
    def delete_app_version(
        self, app_version: mod_types_schema.AppVersion
    ) -> None:
        """
        Deletes an app from the database based on its app_version.

        Args:
            app_version (schema.AppVersion): The unique identifier of the app to be deleted.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self,
        app_versions: Optional[List[mod_types_schema.AppVersion]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Get records from the database.

        Args:
            app_versions: If given, retrieve only the records for the given app versions.
                Otherwise all apps are retrieved.

            offset: Database row offset.

            limit: Limit on rows (records) returned.

        Returns:
            A DatFrame with the records.

            A list of column names that contain feedback results.
        """
        raise NotImplementedError()
