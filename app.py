"""Streamlit UI: Copilot and Dashboard tabs (ARCHITECTURE.md §8).

Run: streamlit run app.py
Evaluation is command-line only: python scripts/run_eval.py
"""

import pandas as pd
import streamlit as st

from copilot import config, data_agent, orchestrator, rag
from copilot.data_store import before_anchor, load_table
from copilot.schemas import CopilotResponse

PERSONAS = ["Cashier", "Store Manager", "Inventory Associate", "Regional Manager", "Customer Service"]
PRIORITY_COLOURS = {"P1": "red", "P2": "orange", "P3": "blue", "P4": "gray"}
# One example per scenario type, written by us (not the public test prompts).
EXAMPLES = {
    "Card reader": "Card reader on register 6 at ST010 shows error 05; the other counters are fine.",
    "Store outage": "No register at ST019 can take card or UPI payments since opening.",
    "Stock-out": "Tennis Ball Pack is out of stock at ST005 and the academy needs it this week. Likely cause?",
    "Price mismatch": "Shelf tag for Goalkeeper Gloves at ST018 shows INR 649 but the till charges INR 749.",
}
# Refund, safety, analytics and injection have no preset on purpose: they are typed live in the demo.
OPEN_STATUSES = ["Open", "In Progress"]


# Sidebar -> (persona, store hint or None). Also shows the run mode and the anchor time.
def sidebar() -> tuple[str, str | None]:
    st.sidebar.title("Retail Ops Copilot")
    persona = st.sidebar.selectbox("Persona", PERSONAS)
    store = st.sidebar.text_input("Store id (optional)", placeholder="ST002").strip().upper()
    mode = "MOCK (no API calls)" if config.MOCK_LLM else f"LIVE ({config.OPENAI_MODEL})"
    st.sidebar.caption(f"Mode: {mode}")
    st.sidebar.caption(f"Anchor time: {config.ANCHOR} IST")
    st.sidebar.caption(f"Priority mode: {config.PRIORITY_MODE}")
    return persona, store or None


# Response -> category / priority / confidence badges and the escalation banner.
def show_header(response: CopilotResponse) -> None:
    category_col, priority_col, confidence_col = st.columns([3, 1, 1])
    category_col.metric("Category", response.category)
    colour = PRIORITY_COLOURS[response.priority]
    priority_col.markdown(f"**Priority**  \n:{colour}[**{response.priority}**]")
    confidence_col.metric("Confidence", f"{response.confidence:.2f}")
    escalation = response.escalation
    if escalation.required:
        approval = " · human approval required" if escalation.human_approval_required else ""
        st.error(f"Escalate to **{escalation.team}** via `{escalation.channel}` · SLA {escalation.sla_minutes} min"
                 f"{approval}  \nReason: {escalation.reason}")
    else:
        st.success("No escalation required.")


# Response -> answer, next actions, follow-up and the "why this priority" line.
def show_answer(response: CopilotResponse) -> None:
    if response.safety_refusal:
        st.warning("Request refused by the safety gate.")
    if response.pii_masked:
        st.info("Personal or payment data in the request was masked before processing.")
    st.subheader("Answer")
    st.write(response.answer)
    if response.next_actions:
        st.markdown("**Next actions**")
        for action in response.next_actions:
            st.markdown(f"- {action}")
    if response.follow_up_question:
        st.info(f"Follow-up: {response.follow_up_question}")
    st.markdown(f"**Why this priority:** `{response.priority_rule_id}` — {response.priority_reason}")


# Response -> evidence, citations (expandable section text), workflow payload and trace id.
def show_evidence(response: CopilotResponse) -> None:
    with st.expander("Evidence (data findings)"):
        st.write("Data used:", ", ".join(response.data_used) or "none")
        st.json(response.data_findings)
    st.markdown("**Citations**")
    for citation in response.citations:
        with st.expander(citation):
            st.markdown(rag.get_section(citation) or "Section text not found.")
    if response.workflow_payload:
        with st.expander(f"Workflow payload ({response.workflow_payload['type']}, mock — nothing was sent)"):
            st.json(response.workflow_payload)
    st.caption(f"Trace id: {response.trace_id}")


# Copilot tab: free-text input first, optional example buttons, run, result.
def copilot_tab(persona: str, store_hint: str | None) -> None:
    st.subheader("What's happening in the store?")
    query_box = st.container()
    st.caption("Or start from an example:")
    example_columns = st.columns(len(EXAMPLES))
    for column, (label, text) in zip(example_columns, EXAMPLES.items()):
        if column.button(label, width="stretch"):
            st.session_state["query"] = text
    # Drawn after the buttons run (so a click can fill it) but placed above them on the page.
    query = query_box.text_area("Type any issue or question", key="query", height=140,
                                placeholder="Describe the issue in your own words... e.g. a customer, "
                                            "a product, a store, a till, or a question about the data.")
    if st.button("Ask copilot", type="primary") and query.strip():
        with st.spinner("Thinking..."):
            st.session_state["response"] = orchestrator.run(query, persona, store_hint)
    response = st.session_state.get("response")
    if response:
        show_header(response)
        show_answer(response)
        show_evidence(response)


# Support cases up to the anchor, joined to the store region (regions are shown as given in store_master).
@st.cache_data
def anchored_cases() -> pd.DataFrame:
    cases = before_anchor(load_table("support_cases"), "created_ts")
    stores = load_table("store_master")[["store_id", "region"]]
    cases = cases.merge(stores, on="store_id", how="left")
    return cases.assign(day=cases["created_ts"].dt.date)


# Dashboard tab: open-case KPIs, priority / category distribution, region trend, vendor quality table.
def dashboard_tab() -> None:
    cases = anchored_cases()
    open_cases = cases[cases["status"].isin(OPEN_STATUSES)]
    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Open / in-progress cases", len(open_cases))
    kpi_columns[1].metric("Open P1", int((open_cases["priority_label"] == "P1").sum()))
    kpi_columns[2].metric("Open P2", int((open_cases["priority_label"] == "P2").sum()))
    left, right = st.columns(2)
    left.markdown("**Open cases by priority**")
    left.bar_chart(open_cases["priority_label"].value_counts().sort_index())
    right.markdown("**Open cases by category**")
    right.bar_chart(open_cases["category_label"].value_counts())
    st.markdown("**Cases per day by region (last 30 days)**")
    recent = cases[cases["created_ts"] > config.ANCHOR - pd.Timedelta(days=config.QUALITY_WINDOW_DAYS)]
    st.line_chart(recent.groupby(["day", "region"]).size().unstack(fill_value=0))
    st.caption("Region values come from store_master.csv as given; some are inconsistent in the source data.")
    st.markdown("**Vendor quality complaints (last 30 days)**")
    st.dataframe(pd.DataFrame(data_agent.vendor_quality_ranking().facts["ranking"]), width="stretch")
    st.caption("Complaint counts are a signal for review, not proof of vendor fault.")


# Page entry point.
def main() -> None:
    st.set_page_config(page_title="Retail Ops Copilot", layout="wide")
    persona, store_hint = sidebar()
    copilot, dashboard = st.tabs(["Copilot", "Dashboard"])
    with copilot:
        copilot_tab(persona, store_hint)
    with dashboard:
        dashboard_tab()


main()
