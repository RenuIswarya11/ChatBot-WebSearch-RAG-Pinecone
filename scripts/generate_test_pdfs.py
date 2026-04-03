"""Generate topic-accurate multi-page sample PDFs for manual testing."""

from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


TOPIC_CONTENT: Dict[str, Tuple[str, List[Tuple[str, List[str]]]]] = {
    "research_methods.pdf": (
        "Research Methods for AI Projects",
        [
            (
                "Section 1: Problem Framing and Hypotheses",
                [
                    "Define the research question as a falsifiable statement.",
                    "Map stakeholders, constraints, and measurable business goals.",
                    "Convert goals into hypotheses with expected effect size.",
                    "Document assumptions on data quality and user behavior.",
                    "Create success metrics before model experimentation starts.",
                    "Plan baseline systems to compare against future models.",
                ],
            ),
            (
                "Section 2: Experimental Design and Ablations",
                [
                    "Use train, validation, and holdout partitions by time.",
                    "Run ablations to isolate contribution of each component.",
                    "Track variance across seeds for reproducibility confidence.",
                    "Include negative controls to detect data leakage early.",
                    "Report confidence intervals, not only point estimates.",
                    "Record all prompts, parameters, and dataset versions.",
                ],
            ),
            (
                "Section 3: Reporting and Decision-Making",
                [
                    "Summarize findings in terms of practical deployment risk.",
                    "Call out failure cases with representative examples.",
                    "Separate statistical significance from business relevance.",
                    "Recommend go, no-go, or iterate decisions with rationale.",
                    "Archive notebooks and scripts for future audits.",
                    "Define next experiments based on unresolved uncertainty.",
                ],
            ),
        ],
    ),
    "rag_design_patterns.pdf": (
        "RAG Design Patterns and Trade-offs",
        [
            (
                "Section 1: Core RAG Architectures",
                [
                    "Single-shot RAG retrieves top-k chunks and answers once.",
                    "Multi-query RAG rewrites the question to improve recall.",
                    "Hybrid RAG combines sparse BM25 with dense vector search.",
                    "Hierarchical RAG retrieves sections, then drills into chunks.",
                    "Agentic RAG can call tools when retrieval confidence is low.",
                    "Each pattern trades latency, precision, and implementation cost.",
                ],
            ),
            (
                "Section 2: Retrieval and Ranking Strategies",
                [
                    "Start with cosine similarity for fast baseline retrieval.",
                    "Add metadata filters to reduce irrelevant document matches.",
                    "Use reranking to improve ordering of semantically close chunks.",
                    "Set minimum relevance thresholds to prevent weak grounding.",
                    "Deduplicate near-identical chunks before final context assembly.",
                    "Tune k-values per question type and document length.",
                ],
            ),
            (
                "Section 3: Failure Modes and Mitigations",
                [
                    "Hallucination risk rises when context is sparse or off-topic.",
                    "Chunk boundary loss can hide critical qualifying statements.",
                    "Outdated corpora require freshness checks or web fallback.",
                    "Citation mismatch occurs if metadata is dropped in pipelines.",
                    "Mitigate with grounded prompts, overlap, and confidence gating.",
                    "Evaluate with question sets covering ambiguity and edge cases.",
                ],
            ),
        ],
    ),
    "vector_search_fundamentals.pdf": (
        "Vector Search Fundamentals",
        [
            (
                "Section 1: Embeddings and Distance Metrics",
                [
                    "Embeddings map text into numeric vectors in latent space.",
                    "Cosine similarity compares direction, not absolute magnitude.",
                    "Dot product can favor long vectors in some settings.",
                    "Euclidean distance works when embedding scales are consistent.",
                    "Model choice changes cluster structure and retrieval behavior.",
                    "Normalize vectors when metric assumptions require it.",
                ],
            ),
            (
                "Section 2: Indexing and Querying",
                [
                    "Approximate nearest neighbor indexes optimize latency at scale.",
                    "Namespace isolation supports multi-tenant search safely.",
                    "Metadata filtering narrows candidate space before scoring.",
                    "Top-k controls recall but can increase noise in prompts.",
                    "Batch upserts reduce network overhead and improve throughput.",
                    "Periodic re-indexing is needed after corpus updates.",
                ],
            ),
            (
                "Section 3: Practical Tuning Guidelines",
                [
                    "Increase chunk size when concepts span multiple sentences.",
                    "Increase overlap to preserve context around boundaries.",
                    "Lower similarity threshold to improve recall for vague queries.",
                    "Raise threshold when false positives dominate retrieved context.",
                    "Measure latency separately for embedding and vector lookup.",
                    "Use evaluation sets to tune ranking and threshold values.",
                ],
            ),
        ],
    ),
    "agent_routing_strategies.pdf": (
        "Agent Routing Strategies",
        [
            (
                "Section 1: Route Selection Policies",
                [
                    "Rule-based routing uses deterministic thresholds and intents.",
                    "Classifier routing predicts the best tool for each query.",
                    "Confidence routing decides between documents and web search.",
                    "Cost-aware routing chooses cheaper tools for simple requests.",
                    "Safety routing blocks risky actions unless strict criteria pass.",
                    "Fallback chains improve robustness when primary tools fail.",
                ],
            ),
            (
                "Section 2: Signals Used for Routing",
                [
                    "Relevance score from retriever indicates document fit.",
                    "Keyword overlap catches topic mismatch despite high similarity.",
                    "Conversation state resolves pronouns and follow-up references.",
                    "Tool health checks prevent routing to unavailable services.",
                    "Quota signals avoid tools that exceed request limits.",
                    "Latency budgets prioritize faster routes under load.",
                ],
            ),
            (
                "Section 3: Observability and Guardrails",
                [
                    "Log route decisions with input features and outcomes.",
                    "Track route accuracy against human-labeled expectations.",
                    "Audit tool usage for cost spikes and error patterns.",
                    "Add user-facing notes when fallback route is activated.",
                    "Create alarms for sustained tool failure rates.",
                    "Continuously refine policy thresholds using production data.",
                ],
            ),
        ],
    ),
    "evaluation_groundedness.pdf": (
        "Evaluation and Groundedness",
        [
            (
                "Section 1: Groundedness Definitions",
                [
                    "Grounded answers must be supported by cited evidence.",
                    "Faithfulness measures whether output matches source context.",
                    "Answer relevance checks alignment with user intent.",
                    "Citation quality requires readable and precise references.",
                    "Unsupported claims should trigger uncertainty statements.",
                    "Groundedness is different from fluency or style quality.",
                ],
            ),
            (
                "Section 2: Evaluation Methodology",
                [
                    "Build benchmark questions with known supporting passages.",
                    "Label expected route: docs-only, web-fallback, or abstain.",
                    "Score retrieval hit rate before judging generation quality.",
                    "Use pairwise review for ambiguous factual questions.",
                    "Track hallucination rate across domains and query types.",
                    "Report metrics with confidence intervals over multiple runs.",
                ],
            ),
            (
                "Section 3: Production Quality Controls",
                [
                    "Add source-required prompts to reduce unsupported claims.",
                    "Reject answers when no high-confidence evidence exists.",
                    "Display citations and route metadata in the UI.",
                    "Run periodic regression tests on curated query sets.",
                    "Use user feedback loops to catch missed grounding issues.",
                    "Document known limits and escalation paths for failures.",
                ],
            ),
        ],
    ),
    "deployment_playbook.pdf": (
        "Deployment Playbook for Streamlit Apps",
        [
            (
                "Section 1: Environment and Secrets Management",
                [
                    "Store API keys in deployment secrets, not source control.",
                    "Validate required variables at app startup with clear errors.",
                    "Use separate keys for development and production environments.",
                    "Pin dependency versions to reduce deployment drift.",
                    "Document minimum system requirements in the README.",
                    "Add health checks to verify external service availability.",
                ],
            ),
            (
                "Section 2: Performance and Reliability",
                [
                    "Use lazy loading for heavy models to speed first paint.",
                    "Cache static resources to reduce repeated network calls.",
                    "Handle timeout errors with actionable retry messages.",
                    "Limit upload size and page count to control compute cost.",
                    "Batch vector upserts and monitor request latency.",
                    "Implement graceful fallback when external tools fail.",
                ],
            ),
            (
                "Section 3: Release and Monitoring",
                [
                    "Deploy from main branch with reproducible build steps.",
                    "Track app errors, tool failures, and response times.",
                    "Define rollback steps for failed releases.",
                    "Use smoke tests for upload, retrieval, and fallback paths.",
                    "Review usage analytics for feature and cost optimization.",
                    "Keep a changelog summarizing user-visible improvements.",
                ],
            ),
        ],
    ),
    "mlops_observability.pdf": (
        "MLOps Observability for GenAI Apps",
        [
            (
                "Section 1: What to Observe",
                [
                    "Monitor input volume, latency, and token consumption.",
                    "Track retrieval quality metrics and route distribution.",
                    "Capture model errors with categorized failure reasons.",
                    "Measure citation presence and fallback activation rate.",
                    "Log embedding and vector-store throughput over time.",
                    "Watch quota usage for each external provider.",
                ],
            ),
            (
                "Section 2: Debugging and Incident Response",
                [
                    "Correlate user queries with backend request traces.",
                    "Store request IDs for cross-service troubleshooting.",
                    "Create dashboards for success, warning, and error states.",
                    "Add alerts for sudden response-time regressions.",
                    "Include synthetic probes for tool availability checks.",
                    "Run post-incident reviews to prevent repeat failures.",
                ],
            ),
            (
                "Section 3: Continuous Improvement Loops",
                [
                    "Use feedback labels to retrain routing thresholds.",
                    "Analyze missed answers to improve chunking and prompts.",
                    "Track long-term quality trends after each release.",
                    "Prioritize fixes by user impact and frequency.",
                    "Benchmark against baseline runs before deployment.",
                    "Maintain an experimentation log for transparent decisions.",
                ],
            ),
        ],
    ),
}


def draw_wrapped_lines(pdf: canvas.Canvas, lines: List[str], x: int, y: int, width: int) -> None:
    """Draw wrapped lines at a fixed width on the PDF canvas.

    Args:
        pdf (canvas.Canvas): ReportLab canvas used for drawing text.
        lines (List[str]): Ordered text lines to write on the page.
        x (int): Left x-coordinate for text drawing.
        y (int): Starting y-coordinate for text drawing.
        width (int): Maximum text width in points before wrapping.

    Returns:
        None: Text is rendered directly onto the PDF canvas.
    """
    cursor_y = y
    for line in lines:
        wrapped = simpleSplit(line, "Helvetica", 11, width)
        for chunk in wrapped:
            pdf.drawString(x, cursor_y, chunk)
            cursor_y -= 16
        cursor_y -= 2


def main() -> None:
    """Generate seven topic-accurate 3-page PDFs for testing.

    Args:
        None: Uses predefined topics and section content.

    Returns:
        None: Output files are written into `test_documents`.
    """
    output_dir = Path("test_documents")
    output_dir.mkdir(exist_ok=True)

    for filename, (title, pages) in TOPIC_CONTENT.items():
        target_path = output_dir / filename
        pdf = canvas.Canvas(str(target_path), pagesize=LETTER)
        for page_number, (section_title, lines) in enumerate(pages, start=1):
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(72, 740, title)
            pdf.setFont("Helvetica", 11)
            pdf.drawString(72, 718, section_title)
            draw_wrapped_lines(pdf, lines, x=72, y=690, width=460)
            pdf.setFont("Helvetica-Oblique", 10)
            pdf.drawString(72, 72, f"Reference page marker: {filename}, page {page_number}")
            pdf.showPage()
        pdf.save()

    print(f"Generated {len(TOPIC_CONTENT)} topic-accurate PDFs in {output_dir}")


if __name__ == "__main__":
    main()

