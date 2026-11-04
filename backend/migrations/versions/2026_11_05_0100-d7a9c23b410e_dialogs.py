"""dialogs and ordered questions

Revision ID: d7a9c23b410e
Revises: 8a7b66557260
"""

import sqlalchemy as sa
from alembic import op

revision = "d7a9c23b410e"
down_revision = "8a7b66557260"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dialogs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("deid_session_id", sa.Uuid(), nullable=False),
        sa.Column("on_date", sa.Date(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_turn", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["auth.users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("deid_session_id"),
        sa.CheckConstraint("next_turn >= 1", name="dialog_next_turn"),
        schema="assistant",
    )
    op.create_index("ix_dialogs_user_id", "dialogs", ["user_id"], schema="assistant")
    op.add_column("questions", sa.Column("dialog_id", sa.Uuid(), nullable=True), schema="assistant")
    op.add_column(
        "questions", sa.Column("turn_no", sa.Integer(), nullable=True), schema="assistant"
    )
    op.create_foreign_key(
        "fk_questions_dialog_id_dialogs",
        "questions",
        "dialogs",
        ["dialog_id"],
        ["id"],
        source_schema="assistant",
        referent_schema="assistant",
        ondelete="CASCADE",
    )
    op.create_index("ix_questions_dialog_id", "questions", ["dialog_id"], schema="assistant")
    op.create_unique_constraint(
        "uq_question_dialog_turn", "questions", ["dialog_id", "turn_no"], schema="assistant"
    )
    op.create_check_constraint(
        "question_dialog_turn",
        "questions",
        "(dialog_id IS NULL AND turn_no IS NULL) OR "
        "(dialog_id IS NOT NULL AND turn_no IS NOT NULL AND turn_no > 0)",
        schema="assistant",
    )


def downgrade() -> None:
    # Сохраняем существующие обезличенные сообщения как отдельные вопросы.
    op.drop_constraint("question_dialog_turn", "questions", schema="assistant", type_="check")
    op.drop_constraint("uq_question_dialog_turn", "questions", schema="assistant", type_="unique")
    op.drop_constraint(
        "fk_questions_dialog_id_dialogs", "questions", schema="assistant", type_="foreignkey"
    )
    op.drop_index("ix_questions_dialog_id", table_name="questions", schema="assistant")
    op.drop_column("questions", "turn_no", schema="assistant")
    op.drop_column("questions", "dialog_id", schema="assistant")
    op.drop_index("ix_dialogs_user_id", table_name="dialogs", schema="assistant")
    op.drop_table("dialogs", schema="assistant")
