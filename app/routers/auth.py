from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.user import Token, UserLogin, UserRegister, UserResponse
from app.services.auth_service import authenticate_user, register_user
from app.utils.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册新用户",
)
async def register(data: UserRegister, db: AsyncSession = Depends(get_db)):
    return await register_user(db, data)


@router.post("/login", response_model=Token, summary="登录获取 token")
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    access_token = await authenticate_user(db, data)
    return Token(access_token=access_token)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前登录用户信息（需鉴权）",
)
async def read_me(current_user: User = Depends(get_current_user)):
    return current_user
