"""鉴权工具单元测试：密码哈希 + JWT 编解码。

不碰 DB/网络，纯依赖 passlib/jose。覆盖：
哈希不等于明文、校验成功/失败、JWT 往返、篡改/过期/垃圾 token。
"""

from datetime import datetime, timedelta, timezone

from jose import jwt

from app.config import settings
from app.utils.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHash:
    def test_hash_differs_from_plain(self):
        # 哈希结果绝不等于明文
        hashed = hash_password("secret123")
        assert hashed != "secret123"

    def test_hash_is_salted_non_deterministic(self):
        # 同一密码两次哈希应不同（bcrypt 自带随机盐）
        assert hash_password("secret123") != hash_password("secret123")

    def test_verify_correct_password(self):
        hashed = hash_password("secret123")
        assert verify_password("secret123", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("secret123")
        assert verify_password("wrong-password", hashed) is False


class TestJWT:
    def test_roundtrip_returns_subject(self):
        # Arrange / Act
        token = create_access_token("user@test.com")
        subject = decode_access_token(token)
        # Assert
        assert subject == "user@test.com"

    def test_garbage_token_returns_none(self):
        assert decode_access_token("not.a.jwt") is None

    def test_wrong_secret_returns_none(self):
        # 用错误密钥签发的 token，校验应失败
        forged = jwt.encode(
            {"sub": "attacker@test.com"}, "wrong-secret", algorithm=settings.algorithm
        )
        assert decode_access_token(forged) is None

    def test_expired_token_returns_none(self):
        # 手动签一个已过期的 token
        expired_payload = {
            "sub": "user@test.com",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        }
        expired = jwt.encode(
            expired_payload, settings.secret_key, algorithm=settings.algorithm
        )
        assert decode_access_token(expired) is None
