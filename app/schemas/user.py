from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """注册请求体"""

    email: EmailStr
    password: str = Field(min_length=6, max_length=72)


class UserLogin(BaseModel):
    """登录请求体"""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """返回给前端的用户信息（不含密码）"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


class Token(BaseModel):
    """登录成功返回的 token"""

    access_token: str
    token_type: str = "bearer"
