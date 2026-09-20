from kaggle_vllm_nebius.agent.audit import AuditTrail


def test_audit_redacts_secrets_without_redacting_token_metrics():
    audit = AuditTrail()
    audit.add(
        "test",
        api_key="private-value",
        nested={"access_token": "private-value", "output_tokens": 512},
    )
    assert audit.events[0]["api_key"] == "[REDACTED]"
    assert audit.events[0]["nested"]["access_token"] == "[REDACTED]"
    assert audit.events[0]["nested"]["output_tokens"] == 512
