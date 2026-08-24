"""Internal Streamlit rollback/QA surface over the canonical query service.

This module intentionally contains no planner or analytics dispatch logic.  It
renders the same AnswerEnvelope returned by FastAPI and exists for one release
as an operator-only rollback and inspection surface.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import streamlit as st

from src.api.server import _build_state
from src.query.models import QueryRequest


@st.cache_resource(show_spinner="Loading the validated Eagle Eye runtime…")
def _runtime() -> dict[str, Any]:
    return _build_state()


def _render_dataset(dataset: Any) -> None:
    labels = {column.field: column.label for column in dataset.columns}
    frame = pd.DataFrame(dataset.rows).rename(columns=labels)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Eagle Eye QA",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #07131f; color: #e8f0f6; }
        [data-testid="stSidebar"] { background: #091b2a; }
        .ee-status { border: 1px solid #17384b; border-radius: 10px; padding: .8rem 1rem; }
        .ee-status strong { color: #5eead4; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    state = _runtime()
    service = state["query_service"]
    capabilities = service.capabilities()
    freshness = capabilities.get("freshness", {})
    conversation_id = st.session_state.setdefault(
        "canonical_qa_conversation_id", f"streamlit_qa_{uuid.uuid4().hex[:16]}"
    )

    with st.sidebar:
        st.title("Eagle Eye")
        st.caption("INTERNAL CANONICAL QA")
        st.markdown("**New Analysis**")
        st.markdown("**Evidence and lineage**")
        st.markdown("**Carbon Emissions**")
        st.divider()
        st.markdown(
            f"<div class='ee-status'><strong>Historical data</strong><br>"
            f"Validated through {freshness.get('data_to') or 'unavailable'}<br>"
            "Not live operational intelligence.</div>",
            unsafe_allow_html=True,
        )

    st.title("Canonical response inspector")
    st.caption(
        "Rollback/QA only. This surface calls the same context resolver, planner, "
        "validator, executor, fact validator, and envelope builder as `/api/v2/query`."
    )
    with st.form("canonical_query", clear_on_submit=False):
        question = st.text_area(
            "Analysis request",
            placeholder="Plot daily arrivals at Gothenburg for March 2022.",
            height=90,
        )
        submitted = st.form_submit_button("Run canonical analysis", type="primary")

    if not submitted:
        st.info("Enter an analysis request to inspect its validated answer envelope.")
        return
    if not question.strip():
        st.warning("Enter a request before running the analysis.")
        return

    with st.spinner("Validating the request and data scope…"):
        envelope = service.query(
            QueryRequest(question=question, conversation_id=conversation_id)
        )

    st.subheader("Intelligence brief")
    state_col, confidence_col, freshness_col = st.columns(3)
    state_col.metric("Result state", envelope.state.value)
    confidence_col.metric("Confidence", envelope.confidence)
    freshness_col.metric("Data through", envelope.freshness.data_to or "Unavailable")
    st.write(envelope.answer)
    for caveat in envelope.caveats:
        st.caption(f"Caveat: {caveat}")

    st.subheader("Validated visualization contract")
    for visualization in envelope.visualizations:
        st.markdown(f"**{visualization.title}** — `{visualization.kind}`")
        st.caption(visualization.accessible_summary)
        st.json(visualization.model_dump(mode="json"), expanded=False)

    st.subheader("View data")
    if envelope.datasets:
        dataset_tabs = st.tabs([dataset.id for dataset in envelope.datasets])
        for tab, dataset in zip(dataset_tabs, envelope.datasets):
            with tab:
                _render_dataset(dataset)
    else:
        st.info("This result has no validated dataset rows.")

    evidence_tab, facts_tab, trace_tab = st.tabs(["Evidence", "Immutable facts", "Developer trace"])
    with evidence_tab:
        if not envelope.evidence:
            st.caption("No external evidence is needed for this computed dataset result.")
        for item in envelope.evidence:
            st.markdown(f"**{item.title}**")
            if item.excerpt:
                st.write(item.excerpt)
            if item.url:
                st.link_button("Open source", item.url)
    with facts_tab:
        st.json([fact.model_dump(mode="json") for fact in envelope.facts], expanded=False)
    with trace_tab:
        st.json(envelope.trace.model_dump(mode="json"), expanded=False)


if __name__ == "__main__":
    main()
