"""
Security regression tests for the knowledge server.

Run with:  pytest -q
No network access or OpenRouter key is needed.
"""
import importlib
import os
import sys
import tempfile

import pytest

ADMIN_EMAIL = "manager@test.local"
ADMIN_PASSWORD = "Test-Password-123"
API_KEY = "test-api-key-0123456789"


@pytest.fixture(scope="module")
def app_module():
    tmp = tempfile.mkdtemp(prefix="kms-test-")
    os.environ.update({
        "SECRET_KEY": "test-secret",
        "ADMIN_EMAIL": ADMIN_EMAIL,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        "CHROMADB_API_KEY": API_KEY,
        "CHROMADB_PATH": os.path.join(tmp, "chromadb"),
        "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "test.db"),
        "OPENROUTER_API_KEY": "",
    })
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    module = importlib.import_module("app")
    module.app.config.update(TESTING=True, RATELIMIT_ENABLED=False)
    module.limiter.enabled = False
    yield module


@pytest.fixture()
def client(app_module):
    with app_module.app.test_client() as c:
        yield c


def login(client, password=ADMIN_PASSWORD, next_url=None):
    url = "/login" + (f"?next={next_url}" if next_url else "")
    return client.post(url, data={"email": ADMIN_EMAIL, "password": password})


# ---------- credentials ----------

def test_no_hardcoded_default_admin_password():
    with open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8") as f:
        source = f.read()
    assert "admin123" not in source
    assert "change-this-secret-key" not in source


def test_admin_created_from_environment(client):
    assert login(client).status_code == 302
    assert client.get("/dashboard").status_code == 200


def test_wrong_password_rejected(client):
    resp = login(client, password="admin123")
    assert resp.status_code == 200  # login page re-rendered, no redirect
    assert client.get("/dashboard").status_code == 302


def test_generated_password_when_env_missing(app_module):
    app, db, User = app_module.app, app_module.db, app_module.User
    with app.app_context():
        User.query.filter_by(email=ADMIN_EMAIL).delete()
        db.session.commit()
    app.config["ADMIN_PASSWORD"] = ""
    try:
        app_module.init_app()
        with app.app_context():
            admin = User.query.filter_by(email=ADMIN_EMAIL).first()
            assert admin is not None
            assert not admin.check_password("")
            assert not admin.check_password("admin123")
            # restore the known password for the other tests
            admin.set_password(ADMIN_PASSWORD)
            db.session.commit()
    finally:
        app.config["ADMIN_PASSWORD"] = ADMIN_PASSWORD


# ---------- open redirect ----------

@pytest.mark.parametrize("target", ["https://evil.example", "//evil.example", "/\\evil.example"])
def test_login_ignores_external_next(client, target):
    resp = login(client, next_url=target)
    assert resp.status_code == 302
    assert "evil.example" not in resp.headers["Location"]


def test_login_keeps_local_next(client):
    resp = login(client, next_url="/files")
    assert resp.headers["Location"].endswith("/files")


# ---------- API key ----------

def test_search_requires_key(client):
    resp = client.post("/search", json={"query": "test"})
    assert resp.status_code == 401


def test_search_rejects_wrong_key(client):
    resp = client.post("/search", json={"query": "test"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


@pytest.mark.parametrize("headers", [
    {"X-API-Key": API_KEY},
    {"Authorization": f"Bearer {API_KEY}"},
])
def test_search_accepts_configured_key(client, headers):
    resp = client.post("/search", json={"query": "test"}, headers=headers)
    # Authentication passes; without an embeddings key the service itself is unavailable.
    assert resp.status_code != 401


def test_query_string_key_not_accepted(client):
    resp = client.post(f"/search?api_key={API_KEY}", json={"query": "test"})
    assert resp.status_code == 401


def test_api_disabled_when_server_key_missing(client, app_module):
    app_module.app.config["CHROMADB_API_KEY"] = ""
    try:
        resp = client.get("/api/stats", headers={"X-API-Key": "anything"})
        assert resp.status_code == 503
    finally:
        app_module.app.config["CHROMADB_API_KEY"] = API_KEY


def test_stats_with_key(client):
    resp = client.get("/api/stats", headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_health_check_is_public(client):
    assert client.get("/api/health-check").status_code == 200
