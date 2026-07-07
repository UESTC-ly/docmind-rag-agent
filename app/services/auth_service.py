from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserLogin, UserRegister
from app.utils.security import (
    create_access_token,
    hash_password,
    verify_password,
)


async def register_user(db: AsyncSession, data: UserRegister) -> User:
    """注册新用户，email 重复则报错。"""
    result = await db.execute(select(User).where(User.email == data.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该邮箱已被注册",
        )

    user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    await db.flush()  # 刷新以拿到自增 id
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, data: UserLogin) -> str:
    """校验邮箱密码，成功返回 JWT。"""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
        )

    return create_access_token(subject=user.email)
