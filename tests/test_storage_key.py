from data_lake.storage import _classify_key
import base64, json


def _b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def make_jwt(role: str) -> str:
    header = _b64({"alg": "HS256"})
    payload = _b64({"role": role})
    return f"{header}.{payload}.sig"

def test_classify_service_role():
    token = make_jwt("service_role")
    assert _classify_key(token) == "service_role"

def test_classify_publishable():
    assert _classify_key("sb_test") == "publishable"

def test_classify_not_jwt():
    assert _classify_key("notajwt") == "not_jwt"

def test_classify_invalid_jwt():
    assert _classify_key("a.b.c") == "invalid_jwt"


def test_classify_invalid_header():
    payload = base64.urlsafe_b64encode(json.dumps({"role": "service_role"}).encode()).decode().rstrip("=")
    token = f"x.{payload}.y"  # invalid header segment
    assert _classify_key(token) == "invalid_jwt"
