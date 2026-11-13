import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ForgeDataset.normalization.schemas import Chunk, QAPair, QACategory, QAStatus, Evidence
from ForgeDataset.evaluation.numeric_validator import NumericValidator
from ForgeDataset.evaluation.deduplication import SemanticDeduplicator
from ForgeDataset.evaluation.quality_gate import QualityGate


def test_numeric_validator_pass():
    validator = NumericValidator()
    source_context = "In 2023, total revenue reached 450 million with a growth rate of 15%."
    passed, flags = validator.validate("Revenue was 450 million in 2023 with 15% growth.", source_context)
    assert passed is True
    assert len(flags) == 0


def test_numeric_validator_hallucinated_number():
    validator = NumericValidator()
    source_context = "In 2023, total revenue reached 450 million with a growth rate of 15%."
    failed, flags = validator.validate("Revenue was 999 million.", source_context)
    assert failed is False
    assert any("999" in f for f in flags)


def test_numeric_validator_no_numbers_passes():
    """Answers with no numbers bypass numeric validation."""
    validator = NumericValidator()
    source = "Transformers use attention mechanisms to relate positions."
    passed, flags = validator.validate("Transformers use self-attention mechanisms.", source)
    assert passed is True


def test_deduplicator_first_entry_not_duplicate():
    dedup = SemanticDeduplicator()
    qa1 = QAPair(
        chunk_id="c1", doc_id="d1",
        qa_type=QACategory.DEFINITION,
        question="What is an activation function in deep learning?",
        answer="It introduces non-linearity.",
        evidence=[Evidence(text="An activation function introduces non-linearity.")]
    )
    is_dup, score, reason = dedup.check_and_record(qa1)
    assert is_dup is False
    assert score == 0.0


def test_deduplicator_catches_exact_duplicate():
    dedup = SemanticDeduplicator()
    qa1 = QAPair(
        chunk_id="c1", doc_id="d1",
        qa_type=QACategory.DEFINITION,
        question="What is an activation function in deep learning?",
        answer="It introduces non-linearity.",
        evidence=[Evidence(text="An activation function introduces non-linearity.")]
    )
    dedup.check_and_record(qa1)
    is_dup, score, reason = dedup.check_and_record(qa1)
    assert is_dup is True
    assert "Exact" in reason


def test_quality_gate_non_negotiable_evidence_rule():
    """Non-negotiable: empty evidence must always produce REJECTED status."""
    gate = QualityGate()
    chunk = Chunk(
        doc_id="d1", source_filename="test.md",
        content="Support Vector Machines find the optimal hyperplane maximizing the margin.",
        token_count=20, heading_path=["ML", "SVM"]
    )
    qa = QAPair(
        chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
        qa_type=QACategory.APPLICATION,
        question="What is SVM?",
        answer="A machine learning model.",
        evidence=[]  # <-- Non-negotiable rule violation
    )
    result = gate.evaluate(qa, chunk)
    assert result.status == QAStatus.REJECTED
    assert "Non-negotiable rule violation" in result.rejection_reason


def test_quality_gate_approves_fully_grounded_qa():
    """Fully traceable, verbatim-evidence-linked QA should be auto-approved."""
    gate = QualityGate()
    chunk = Chunk(
        doc_id="d1", source_filename="test.md",
        content="Support Vector Machines find the optimal hyperplane maximizing the margin between classes.",
        token_count=20, heading_path=["ML", "SVM"]
    )
    qa = QAPair(
        chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
        qa_type=QACategory.DEFINITION,
        question="What do Support Vector Machines find?",
        answer="Support Vector Machines find the optimal hyperplane maximizing the margin between classes.",
        evidence=[Evidence(text="Support Vector Machines find the optimal hyperplane maximizing the margin between classes.")]
    )
    result = gate.evaluate(qa, chunk)
    assert result.status == QAStatus.APPROVED
    assert result.groundedness_score >= 0.85
    assert result.verification.grounded is True
    assert result.verification.numeric_passed is True


def test_quality_gate_untraceable_evidence_rejected():
    """Evidence not found in source chunk triggers traceability failure."""
    gate = QualityGate()
    chunk = Chunk(
        doc_id="d1", source_filename="test.md",
        content="Gradient descent iteratively minimizes the loss function.",
        token_count=15, heading_path=["Optimization"]
    )
    qa = QAPair(
        chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
        qa_type=QACategory.CONCEPTUAL,
        question="How does Adam optimizer work?",
        answer="Adam optimizer uses momentum and adaptive learning rates.",
        evidence=[Evidence(text="Completely fabricated evidence not in source")]
    )
    result = gate.evaluate(qa, chunk)
    assert result.status == QAStatus.REJECTED
    assert result.groundedness_score < 0.65
