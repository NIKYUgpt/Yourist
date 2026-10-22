"""Служебные команды.

python -m app.cli create-admin --email admin@example.com
"""

import argparse
import asyncio
import getpass
import sys

from email_validator import EmailNotValidError, validate_email
from sqlalchemy import select

from app.core.security import hash_password
from app.core.time import utcnow
from app.db.session import SessionLocal, engine
from app.modules.auth.models import Role, User


async def create_admin(email: str) -> None:
    async with SessionLocal() as session:
        user = await session.scalar(select(User).where(User.email == email))
        now = utcnow()
        if user is not None:
            # существующего пользователя просто повышаем, пароль не трогаем
            user.role = Role.ADMIN
            user.is_active = True
            user.email_verified_at = user.email_verified_at or now
            print(f"Пользователь {email} назначен администратором")
        else:
            password = getpass.getpass("Пароль: ")
            if len(password) < 8:
                sys.exit("Пароль должен быть не короче 8 символов")
            if password != getpass.getpass("Повторите пароль: "):
                sys.exit("Пароли не совпадают")
            session.add(
                User(
                    email=email,
                    password_hash=hash_password(password),
                    role=Role.ADMIN,
                    email_verified_at=now,
                    pd_consent_at=now,
                )
            )
            print(f"Администратор {email} создан")
        await session.commit()
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser(
        "create-admin", help="создать администратора или повысить пользователя"
    )
    admin.add_argument("--email", required=True)
    args = parser.parse_args()

    if args.command == "create-admin":
        try:
            email = validate_email(args.email, check_deliverability=False).normalized.lower()
        except EmailNotValidError as exc:
            sys.exit(f"Некорректный email: {exc}")
        asyncio.run(create_admin(email))


if __name__ == "__main__":
    main()
