import base64
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_lake.storage import _classify_key

def make_jwt(role: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
    return f"x.{payload}.y"

def test_classify_service_role():
    token = make_jwt("service_role")
    assert _classify_key(token) == "service_role"

def test_classify_publishable():
    assert _classify_key("sb_test") == "publishable"

def test_classify_not_jwt():
    assert _classify_key("notajwt") == "not_jwt"

def test_classify_invalid_jwt():
    assert _classify_key("a.b.c") == "invalid_jwt"
