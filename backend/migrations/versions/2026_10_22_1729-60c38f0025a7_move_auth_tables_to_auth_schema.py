"""move auth tables to auth schema

Revision ID: 60c38f0025a7
Revises: e4e4c5650497
Create Date: 2026-10-22 17:29:13.305650

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "60c38f0025a7"
down_revision: str | Sequence[str] | None = "e4e4c5650497"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("users", "refresh_tokens")


def upgrade() -> None:
    """Upgrade schema."""
    # индексы, ограничения и внешние ключи переезжают вместе с таблицами
    op.execute("CREATE SCHEMA IF NOT EXISTS auth")
    for table in TABLES:
        op.execute(f"ALTER TABLE public.{table} SET SCHEMA auth")


def downgrade() -> None:
    """Downgrade schema."""
    for table in TABLES:
        op.execute(f"ALTER TABLE auth.{table} SET SCHEMA public")
    op.execute("DROP SCHEMA auth")
