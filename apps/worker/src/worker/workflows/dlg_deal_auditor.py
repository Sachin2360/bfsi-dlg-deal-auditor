from datetime import timedelta
from typing import Literal

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    ChatAssistantWorkflowOutput,
    ChatCompletionRequest,
    SystemMessage,
    TextOutput,
    UserMessage,
    chat_parse_to_model,
    mistralai_chat_complete,
)
from pydantic import BaseModel, Field

DLG_CAP_PCT = 5.0
PROHIBITED_CHANNELS = {"nbfc-p2p", "credit card", "revolving credit", "ncgtc"}
MEMO_MODEL = "mistral-large-latest"
JUDGE_MODEL = "mistral-large-latest"

RBI_DLG_FAQ_RULES = """Authoritative RBI DLG FAQ rules (use ONLY these, cite exact Q numbers):
- Q.1 & Q.2: DLG cover shall not exceed 5% of the total amount disbursed out of the upfront DLG set.
- Q.7: DLG is not permitted on NBFC-P2P platforms.
- Q.8: DLG provider declaration requires statutory auditor certification.
- Q.9/Q.10/Q.12: DLG is prohibited on credit cards, revolving credit, and NCGTC loans.
- Q.11: an RE providing DLG must deduct the full outstanding DLG amount from its capital."""


class DlgDealInput(BaseModel):
    deal_name: str = "FinTech Q3 Digital Loan Pool"
    disbursed_pool_crore: float = 200.0
    proposed_dlg_crore: float = 12.0
    channel_type: str = "NBFC-P2P"
    provider_is_re: bool = True


class DlgCheckResult(BaseModel):
    deal_name: str
    disbursed_pool_crore: float
    proposed_dlg_crore: float
    channel_type: str
    provider_is_re: bool
    dlg_pct: float
    cap_breached: bool
    prohibited_channel: bool


class FaithfulnessAudit(BaseModel):
    faithfulness_score: int = Field(ge=1, le=5)
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_extrapolations: list[str] = Field(default_factory=list)
    judge_verdict: Literal["PASS", "FAIL"]


class DlgAuditResult(BaseModel):
    deal_metrics: DlgCheckResult
    compliance_memo: str
    faithfulness_audit: FaithfulnessAudit


def _compute_dlg_check(deal: DlgDealInput) -> DlgCheckResult:
    dlg_pct = (deal.proposed_dlg_crore / deal.disbursed_pool_crore) * 100
    channel = deal.channel_type.strip().lower()
    return DlgCheckResult(
        deal_name=deal.deal_name,
        disbursed_pool_crore=deal.disbursed_pool_crore,
        proposed_dlg_crore=deal.proposed_dlg_crore,
        channel_type=deal.channel_type,
        provider_is_re=deal.provider_is_re,
        dlg_pct=dlg_pct,
        cap_breached=dlg_pct > DLG_CAP_PCT,
        prohibited_channel=channel in PROHIBITED_CHANNELS,
    )


@workflows.activity(start_to_close_timeout=timedelta(seconds=60))
async def calculate_dlg_check(deal: DlgDealInput) -> DlgCheckResult:
    return _compute_dlg_check(deal)


@workflows.activity(start_to_close_timeout=timedelta(seconds=300))
async def draft_rbi_memo(deal: DlgDealInput, metrics: DlgCheckResult) -> str:
    request = ChatCompletionRequest(
        model=MEMO_MODEL,
        temperature=0.0,
        top_p=1.0,
        messages=[
            SystemMessage(
                content=(
                    "You are a BFSI compliance officer drafting an RBI Default Loan Guarantee (DLG) "
                    "compliance decision memo. Base every determination ONLY on the authoritative RBI "
                    "DLG FAQ rules provided, cite the exact FAQ Q number for each determination, and do "
                    "not extrapolate beyond those rules."
                )
            ),
            UserMessage(
                content=(
                    f"Deal inputs:\n{deal.model_dump_json(indent=2)}\n\n"
                    f"Activity 1 (calculate_dlg_check) math:\n{metrics.model_dump_json(indent=2)}\n\n"
                    f"{RBI_DLG_FAQ_RULES}\n\n"
                    "Write a concise compliance decision memo for this deal. For each finding, cite the "
                    "exact RBI FAQ Q number it rests on (e.g. Q.1/Q.2, Q.7, Q.8, Q.9/Q.10/Q.12, Q.11). "
                    "State the decision explicitly. Do not cite or rely on any rule not listed above."
                )
            ),
        ],
    )
    response = await mistralai_chat_complete(request)
    return response.choices[0].message.content


@workflows.activity(start_to_close_timeout=timedelta(seconds=300))
async def audit_memo_faithfulness(
    deal: DlgDealInput, metrics: DlgCheckResult, memo: str
) -> FaithfulnessAudit:
    request = ChatCompletionRequest(
        model=JUDGE_MODEL,
        temperature=0.0,
        top_p=1.0,
        messages=[
            SystemMessage(
                content=(
                    "You are a strict LLM-as-a-judge auditing a compliance memo for faithfulness. "
                    "Compare the memo ONLY against the authoritative RBI DLG FAQ rules and the exact "
                    "Activity 1 math provided. Return strict JSON matching the required schema: "
                    '{"faithfulness_score": 1-5, "supported_claims": [...], '
                    '"unsupported_extrapolations": [...], "judge_verdict": "PASS" or "FAIL"}. '
                    "verdict is PASS only if every memo claim is supported and the math matches; "
                    "otherwise FAIL. Do not add any field outside the schema."
                )
            ),
            UserMessage(
                content=(
                    f"Deal inputs:\n{deal.model_dump_json(indent=2)}\n\n"
                    f"Activity 1 (calculate_dlg_check) math:\n{metrics.model_dump_json(indent=2)}\n\n"
                    f"{RBI_DLG_FAQ_RULES}\n\n"
                    f"Activity 2 (draft_rbi_memo) memo under audit:\n{memo}\n\n"
                    "Audit the memo strictly against the rules above and the math. List each memo claim "
                    "that is supported, and each claim or inference that is NOT supported "
                    "(extrapolations). Score faithfulness 1 (mostly unfaithful) to 5 (fully faithful)."
                )
            ),
        ],
    )
    return await chat_parse_to_model(FaithfulnessAudit, request)


def _render_markdown(result: DlgAuditResult) -> str:
    m = result.deal_metrics
    a = result.faithfulness_audit
    lines = [
        f"# BFSI DLG Deal Audit — {m.deal_name}",
        "",
        f"- Disbursed pool: {m.disbursed_pool_crore} Cr | Proposed DLG: {m.proposed_dlg_crore} Cr",
        f"- DLG %: **{m.dlg_pct:.2f}%** (RBI cap 5.00%, Q.1/Q.2) → cap breached: **{m.cap_breached}**",
        f"- Channel: {m.channel_type} → prohibited channel: **{m.prohibited_channel}** (Q.7, Q.9/Q.10/Q.12)",
        f"- DLG provider is RE: {m.provider_is_re} (Q.8 certification, Q.11 capital deduction)",
        "",
        "## Compliance memo (RBI DLG FAQ-cited)",
        "",
        result.compliance_memo,
        "",
        "## Faithfulness audit (LLM-as-a-judge)",
        "",
        f"- Score: **{a.faithfulness_score}/5** — verdict: **{a.judge_verdict}**",
        f"- Supported claims: {len(a.supported_claims)} | Unsupported extrapolations: {len(a.unsupported_extrapolations)}",
        "",
        "Full structured result is in `structuredContent` (deal_metrics, compliance_memo, faithfulness_audit).",
    ]
    return "\n".join(lines)


@workflows.workflow.define(
    name="bfsi-dlg-deal-auditor",
    workflow_display_name="BFSI DLG Deal Auditor",
    workflow_description=(
        "Audits a BFSI loan-pool deal against RBI DLG rules: computes the exact DLG percentage and "
        "channel checks, drafts an RBI FAQ-cited compliance memo with a Mistral model, and "
        "LLM-judges the memo's faithfulness to the rules and math. Publishable to Vibe Work."
    ),
)
class BfsiDlgDealAuditorWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: DlgDealInput) -> ChatAssistantWorkflowOutput:
        metrics = await calculate_dlg_check(input)
        memo = await draft_rbi_memo(input, metrics)
        audit = await audit_memo_faithfulness(input, metrics, memo)
        result = DlgAuditResult(
            deal_metrics=metrics,
            compliance_memo=memo,
            faithfulness_audit=audit,
        )
        return ChatAssistantWorkflowOutput(
            content=[TextOutput(text=_render_markdown(result))],
            structuredContent=result.model_dump(mode="json"),
        )
