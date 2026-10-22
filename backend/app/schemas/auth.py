from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    pd_consent: bool = Field(description="Согласие на обработку персональных данных")

    @field_validator("pd_consent")
    @classmethod
    def _consent_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Необходимо согласие на обработку персональных данных")
        return value


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Время жизни access-токена, сек")


class RefreshIn(BaseModel):
    refresh_token: str


class EmailTokenIn(BaseModel):
    token: str


class EmailIn(BaseModel):
    email: EmailStr


class MessageOut(BaseModel):
    detail: str
