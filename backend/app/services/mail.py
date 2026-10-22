import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage
from functools import lru_cache
from typing import Protocol

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Mailer(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...


def _build_message(sender: str, to: str, subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


class ConsoleMailer:
    """Пишет письма в лог и складывает в outbox. Для разработки и тестов."""

    def __init__(self, sender: str = "Yourist <no-reply@localhost>") -> None:
        self.sender = sender
        self.outbox: list[EmailMessage] = []

    async def send(self, to: str, subject: str, body: str) -> None:
        self.outbox.append(_build_message(self.sender, to, subject, body))
        logger.info("Письмо для %s: %s\n%s", to, subject, body)


class SmtpMailer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send(self, to: str, subject: str, body: str) -> None:
        msg = _build_message(self.settings.mail_from, to, subject, body)
        # smtplib синхронный — уводим в поток, чтобы не блокировать event loop
        await asyncio.to_thread(self._send_sync, msg)

    def _send_sync(self, msg: EmailMessage) -> None:
        s = self.settings
        context = ssl.create_default_context()
        if s.smtp_ssl:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, context=context, timeout=15) as smtp:
                smtp.login(s.smtp_user, s.smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
                smtp.starttls(context=context)
                smtp.login(s.smtp_user, s.smtp_password)
                smtp.send_message(msg)


@lru_cache
def get_mailer() -> Mailer:
    settings = get_settings()
    if settings.mail_backend == "smtp":
        return SmtpMailer(settings)
    return ConsoleMailer(settings.mail_from)
