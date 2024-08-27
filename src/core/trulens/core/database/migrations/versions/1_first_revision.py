"""First revision.

Revision ID: 1
Revises:
Create Date: 2023-08-10 23:11:37.405982
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    if prefix is None:
        raise RuntimeError("trulens.table_prefix is not set")

    # ### begin Alembic commands ###
    op.create_table(
        prefix + "apps",
        sa.Column("app_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("app_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("app_id"),
    )
    op.create_table(
        prefix + "feedback_defs",
        sa.Column(
            "feedback_definition_id", sa.VARCHAR(length=256), nullable=False
        ),
        sa.Column("feedback_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("feedback_definition_id"),
    )
    op.create_table(
        prefix + "feedbacks",
        sa.Column("feedback_result_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("record_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column(
            "feedback_definition_id", sa.VARCHAR(length=256), nullable=True
        ),
        sa.Column("last_ts", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("calls_json", sa.Text(), nullable=False),
        sa.Column("result", sa.Float(), nullable=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("cost_json", sa.Text(), nullable=False),
        sa.Column("multi_result", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("feedback_result_id"),
    )
    op.create_table(
        prefix + "records",
        sa.Column("record_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("app_id", sa.VARCHAR(length=256), nullable=False),
        sa.Column("input", sa.Text(), nullable=True),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("record_json", sa.Text(), nullable=False),
        sa.Column("tags", sa.Text(), nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("cost_json", sa.Text(), nullable=False),
        sa.Column("perf_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("record_id"),
    )
    # ### end Alembic commands ###


def downgrade(config) -> None:
    prefix = config.get_main_option("trulens.table_prefix")

    # ### begin Alembic commands ###
    op.drop_table(prefix + "records")
    op.drop_table(prefix + "feedbacks")
    op.drop_table(prefix + "feedback_defs")
    op.drop_table(prefix + "apps")
    # ### end Alembic commands ###
