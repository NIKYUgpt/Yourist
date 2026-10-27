from dataclasses import dataclass

# Виды персональных данных. Метки русские: их видит языковая модель в тексте вопроса
PERSON = "ФИО"
PHONE = "ТЕЛЕФОН"
EMAIL = "EMAIL"
PASSPORT = "ПАСПОРТ"
SNILS = "СНИЛС"
INN = "ИНН"
OGRN = "ОГРН"
CARD = "КАРТА"
ACCOUNT = "СЧЁТ"
ADDRESS = "АДРЕС"
BIRTH_DATE = "ДАТА_РОЖДЕНИЯ"
ORGANIZATION = "ОРГАНИЗАЦИЯ"
PROFILE = "ПРОФИЛЬ"  # ссылки на соцсети, ники

TYPES = (
    PERSON,
    PHONE,
    EMAIL,
    PASSPORT,
    SNILS,
    INN,
    OGRN,
    CARD,
    ACCOUNT,
    ADDRESS,
    BIRTH_DATE,
    ORGANIZATION,
    PROFILE,
)

# откуда находка: правило (регулярное выражение с проверкой), словарь имён, языковая модель
RULE = "rule"
DICTIONARY = "dictionary"
LLM = "llm"


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    type: str
    source: str

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end
