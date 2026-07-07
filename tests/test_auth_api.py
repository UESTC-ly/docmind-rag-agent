"""鉴权路由集成测试：注册 / 登录 / 获取当前用户。

用内存库跑真实 HTTP 请求，验证状态码、鉴权、越权与错误分支。
"""

import pytest

pytestmark = pytest.mark.asyncio


class TestRegister:
    async def test_register_success(self, client):
        resp = await client.post(
            "/auth/register", json={"email": "new@test.com", "password": "pass123"}
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["email"] == "new@test.com"
        assert "id" in body
        assert "hashed_password" not in body  # 绝不泄露哈希

    async def test_register_duplicate_email_rejected(self, client):
        payload = {"email": "dup@test.com", "password": "pass123"}
        await client.post("/auth/register", json=payload)
        resp = await client.post("/auth/register", json=payload)
        assert resp.status_code == 400

    async def test_register_short_password_rejected(self, client):
        # schema 要求密码 >=6 位
        resp = await client.post(
            "/auth/register", json={"email": "x@test.com", "password": "123"}
        )
        assert resp.status_code == 422


class TestLogin:
    async def test_login_returns_token(self, client):
        await client.post(
            "/auth/register", json={"email": "u@test.com", "password": "pass123"}
        )
        resp = await client.post(
            "/auth/login", json={"email": "u@test.com", "password": "pass123"}
        )
        assert resp.status_code == 200
        assert resp.json()["access_token"]

    async def test_login_wrong_password_401(self, client):
        await client.post(
            "/auth/register", json={"email": "u2@test.com", "password": "pass123"}
        )
        resp = await client.post(
            "/auth/login", json={"email": "u2@test.com", "password": "wrong"}
        )
        assert resp.status_code == 401

    async def test_login_nonexistent_user_401(self, client):
        resp = await client.post(
            "/auth/login", json={"email": "ghost@test.com", "password": "pass123"}
        )
        assert resp.status_code == 401


class TestCurrentUser:
    async def test_me_with_valid_token(self, client, registered_user):
        resp = await client.get("/auth/me", headers=registered_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["email"] == registered_user["email"]

    async def test_me_without_token_401(self, client):
        resp = await client.get("/auth/me")
        assert resp.status_code == 401

    async def test_me_with_garbage_token_401(self, client):
        resp = await client.get(
            "/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
        )
        assert resp.status_code == 401
