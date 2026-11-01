import time

import jwt
import pytest
from conftest import JWT_SECRET
from fastapi.testclient import TestClient

from careagents.api.app import create_app


def token(role="member", member_id="SARAH-1001", sub="u1", **extra):
    claims = {"sub": sub, "role": role, "aud": "careagents", "exp": int(time.time()) + 600, **extra}
    if member_id:
        claims["member_id"] = member_id
    return {"Authorization": "Bearer " + jwt.encode(claims, JWT_SECRET, algorithm="HS256")}


@pytest.fixture
async def client(make_container, settings):
    container = await make_container()
    app = create_app(settings, container)
    with TestClient(app) as c:
        yield c


def test_probes(client):
    assert client.get("/livez").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"]["database"] is True


def test_requires_auth(client):
    assert client.post("/v1/chat", json={"question": "hi"}).status_code == 401


def test_rejects_bad_signature(client):
    bad = jwt.encode({"sub": "x", "role": "member", "member_id": "SARAH-1001", "aud": "careagents",
                      "exp": int(time.time()) + 60}, "wrong-secret-wrong-secret-wrong-secret!!", algorithm="HS256")
    r = client.post("/v1/chat", json={"question": "hi"}, headers={"Authorization": f"Bearer {bad}"})
    assert r.status_code == 401


def test_member_chat(client):
    r = client.post("/v1/chat", json={"question": "How much will my knee MRI cost?"}, headers=token())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "answered" and body["case_id"]
    assert r.headers["cache-control"] == "no-store" and r.headers["x-request-id"]


def test_member_cannot_target_other_member(client):
    r = client.post("/v1/chat", json={"question": "my claims", "member_id": "JOHN-2002"}, headers=token())
    assert r.status_code == 403


def test_staff_must_name_member_and_member_must_exist(client):
    staff = token(role="staff", member_id=None)
    assert client.post("/v1/chat", json={"question": "claims"}, headers=staff).status_code == 400
    r = client.post("/v1/chat", json={"question": "claims", "member_id": "NOBODY-1"}, headers=staff)
    assert r.status_code == 404
    r = client.post("/v1/chat", json={"question": "claim status", "member_id": "JOHN-2002"}, headers=staff)
    assert r.status_code == 200


def test_validation_errors_do_not_echo_input(client):
    r = client.post("/v1/chat", json={"question": "x" * 5000}, headers=token())
    assert r.status_code == 422 and "xxxx" not in r.text


def test_admin_endpoints_require_admin(client):
    assert client.post("/v1/admin/ingest", headers=token()).status_code == 403
    r = client.post("/v1/admin/ingest", headers=token(role="admin", member_id=None))
    assert r.status_code == 202 and r.json()["status"] == "completed"
    assert r.json()["stats"]["skipped_unchanged"] == 2  # idempotent: nothing changed since boot


async def test_rate_limit(make_container, settings):
    container = await make_container(rate_limit_per_minute=2)
    app = create_app(container.settings, container)
    with TestClient(app) as c:
        codes = [c.post("/v1/chat", json={"question": "claim status"}, headers=token()).status_code
                 for _ in range(3)]
    assert codes == [200, 200, 429]


def test_metrics_exposed(client):
    client.get("/livez")
    assert "ca_http_requests_total" in client.get("/metrics/").text


@pytest.fixture
async def review_client(make_container):
    from fakes import ScriptedLLM

    container = await make_container(llm=ScriptedLLM(compose=lambda _: "You owe $12,345."), human_review_enabled=True)
    with TestClient(create_app(container.settings, container)) as c:
        yield c


def test_review_flow_over_http(review_client):
    c = review_client
    r = c.post("/v1/chat", json={"question": "How much will my knee MRI cost?"}, headers=token())
    case_id = r.json()["case_id"]
    assert r.json()["outcome"] == "pending_review"

    staff = token(role="staff", member_id=None, sub="staff-9")
    assert c.get("/v1/staff/reviews", headers=token()).status_code == 403  # members can't see the queue
    queue = c.get("/v1/staff/reviews", headers=staff).json()
    assert [q["case_id"] for q in queue] == [case_id] and "question" not in queue[0]
    detail = c.get(f"/v1/staff/reviews/{case_id}", headers=staff).json()
    assert detail["review"]["draft"] == "You owe $12,345." and detail["case"]["outcome"] == "pending_review"

    assert c.post(f"/v1/staff/reviews/{case_id}", json={"action": "edit"}, headers=staff).status_code == 422
    done = c.post(f"/v1/staff/reviews/{case_id}", json={"action": "edit", "answer": "About $440."}, headers=staff)
    assert done.status_code == 200 and done.json()["outcome"] == "answered"
    again = c.post(f"/v1/staff/reviews/{case_id}", json={"action": "approve"}, headers=staff)
    assert again.status_code == 409

    mine = c.get(f"/v1/cases/{case_id}", headers=token())
    assert mine.status_code == 200 and mine.json()["answer"] == "About $440."
    other = c.get(f"/v1/cases/{case_id}", headers=token(member_id="JOHN-2002", sub="john"))
    assert other.status_code == 404


def test_chat_stream_sse(client):
    with client.stream("POST", "/v1/chat/stream", json={"question": "How much will my knee MRI cost?"},
                       headers=token()) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events[0] == "started" and "step" in events and events[-1] == "result"


def test_admin_graph_diagram(client):
    r = client.get("/v1/admin/graph", headers=token(role="admin", member_id=None))
    assert r.status_code == 200 and "human_review" in r.text
