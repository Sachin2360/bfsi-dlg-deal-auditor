import asyncio
from datetime import timedelta

import pytest
from mistralai.workflows.testing import create_test_worker
from worker.workflows import dlg_deal_auditor as mod
from worker.workflows.dlg_deal_auditor import (
    PROHIBITED_CHANNELS,
    BfsiDlgDealAuditorWorkflow,
    DlgDealInput,
    FaithfulnessAudit,
    _compute_dlg_check,
    audit_memo_faithfulness,
    calculate_dlg_check,
    draft_rbi_memo,
)

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

DEFAULT_INPUT = {
    "deal_name": "FinTech Q3 Digital Loan Pool",
    "disbursed_pool_crore": 200.0,
    "proposed_dlg_crore": 12.0,
    "channel_type": "NBFC-P2P",
    "provider_is_re": True,
}


def test_dlg_check_math_default_inputs() -> None:
    result = _compute_dlg_check(DlgDealInput())
    assert result.dlg_pct == 6.0
    assert result.cap_breached is True
    assert result.prohibited_channel is True


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("NBFC-P2P", True),
        ("nbfc-p2p", True),
        ("credit card", True),
        ("Revolving Credit", True),
        ("NCGTC", True),
        ("direct lending", False),
    ],
)
def test_prohibited_channel_matching(channel: str, expected: bool) -> None:
    deal = DlgDealInput(channel_type=channel)
    assert _compute_dlg_check(deal).prohibited_channel is expected
    assert len(PROHIBITED_CHANNELS) == 4


def test_dlg_check_under_cap() -> None:
    deal = DlgDealInput(proposed_dlg_crore=10.0, channel_type="direct lending")
    result = _compute_dlg_check(deal)
    assert result.dlg_pct == 5.0
    assert result.cap_breached is False
    assert result.prohibited_channel is False


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _ChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


FAKE_MEMO = "DECISION: REJECT. Q.1/Q.2 breached (6.0% > 5%); Q.7 prohibits NBFC-P2P; Q.11 capital deduction applies."


async def test_workflow_default_inputs_with_mocked_llm(
    temporal_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_chat_complete(_request) -> _ChatResponse:
        return _ChatResponse(FAKE_MEMO)

    async def fake_parse_to_model(model_class, _request):
        return model_class(
            faithfulness_score=5,
            supported_claims=[
                "Q.1/Q.2 breach at 6.0%",
                "Q.7 NBFC-P2P prohibition",
                "Q.11 RE capital deduction",
            ],
            unsupported_extrapolations=[],
            judge_verdict="PASS",
        )

    monkeypatch.setattr(mod, "mistralai_chat_complete", fake_chat_complete)
    monkeypatch.setattr(mod, "chat_parse_to_model", fake_parse_to_model)

    async with create_test_worker(
        temporal_env,
        workflows=[BfsiDlgDealAuditorWorkflow],
        activities=[calculate_dlg_check, draft_rbi_memo, audit_memo_faithfulness],
    ):
        handle = await temporal_env.client.start_workflow(
            "bfsi-dlg-deal-auditor",
            DEFAULT_INPUT,
            id="test-dlg-deal-auditor",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    structured = result["structuredContent"]
    metrics = structured["deal_metrics"]
    assert metrics["dlg_pct"] == 6.0
    assert metrics["cap_breached"] is True
    assert metrics["prohibited_channel"] is True
    assert structured["compliance_memo"] == FAKE_MEMO
    audit = structured["faithfulness_audit"]
    assert audit["faithfulness_score"] == 5
    assert audit["judge_verdict"] == "PASS"
    assert audit["unsupported_extrapolations"] == []
    assert isinstance(FaithfulnessAudit(**audit), FaithfulnessAudit)
    text = result["content"][0]["text"]
    assert "6.00%" in text
    assert FAKE_MEMO in text
    assert "PASS" in text
