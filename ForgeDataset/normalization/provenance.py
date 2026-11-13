"""Provenance tracking and evidence traceability helpers."""

import hashlib
import re
from typing import Optional, Tuple, List, Dict, Any
from .schemas import Chunk, QAPair, AuditRecord, Evidence, DocumentElement


def compute_file_hash(content: bytes) -> str:
    """Compute deterministic SHA-256 hash for document content."""
    return hashlib.sha256(content).hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 of normalized text."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def locate_evidence_in_content(evidence_text: str, content: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """Locate evidence substring within source content, allowing minor whitespace variations.
    
    Returns (found: bool, start_char: Optional[int], end_char: Optional[int])
    """
    if not evidence_text or not content:
        return False, None, None
    
    clean_ev = evidence_text.strip()
    idx = content.find(clean_ev)
    if idx != -1:
        return True, idx, idx + len(clean_ev)
    
    # Try normalized whitespace search
    pattern = re.escape(re.sub(r'\s+', ' ', clean_ev))
    pattern = pattern.replace(r'\ ', r'\s+')
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return True, match.start(), match.end()
    
    # Try substring fuzzy match if long enough (> 30 chars)
    if len(clean_ev) > 30:
        sub_len = int(len(clean_ev) * 0.75)
        sub = clean_ev[:sub_len]
        idx = content.find(sub)
        if idx != -1:
            return True, idx, idx + len(sub)
            
    return False, None, None


def enrich_qa_evidence_provenance(qa: QAPair, chunk: Chunk) -> QAPair:
    """Enrich QA pair evidence items with exact start/end char offsets within the chunk."""
    enriched_evidences: List[Evidence] = []
    for ev in qa.evidence:
        found, start, end = locate_evidence_in_content(ev.text, chunk.content)
        enriched_evidences.append(Evidence(
            text=ev.text,
            source_element_id=chunk.element_ids[0] if chunk.element_ids else None,
            page=chunk.page_start,
            start_char=start,
            end_char=end
        ))
    qa.evidence = enriched_evidences
    return qa


def create_audit_record(qa: QAPair, chunk: Chunk) -> AuditRecord:
    """Construct an audit record strictly conforming to project requirements:
    
    {
      "sample_id": "qa_000001",
      "source": "database_notes.pdf",
      "page": 24,
      "chunk_id": "chunk_0042",
      "qa_type": "conceptual",
      "evidence": [ { "text": "Supporting source sentence..." } ],
      "verification": { "grounded": true, "status": "approved" }
    }
    """
    ev_list = []
    for ev in qa.evidence:
        ev_dict: Dict[str, Any] = {"text": ev.text}
        if ev.page is not None:
            ev_dict["page"] = ev.page
        if ev.start_char is not None:
            ev_dict["start_char"] = ev.start_char
            ev_dict["end_char"] = ev.end_char
        ev_list.append(ev_dict)

    verif_dict = {
        "grounded": qa.groundedness_score >= 0.65,
        "status": qa.status.value,
        "groundedness_score": round(qa.groundedness_score, 4),
        "numeric_valid": qa.numeric_valid,
        "generator_model": qa.generator_model,
        "evaluator_model": qa.evaluator_model
    }
    if qa.verification:
        verif_dict.update({
            "flags": qa.verification.flags,
            "reviewer_notes": qa.verification.reviewer_notes
        })

    return AuditRecord(
        sample_id=qa.qa_id,
        source=chunk.source_filename,
        page=chunk.page_start,
        chunk_id=chunk.chunk_id,
        qa_type=qa.qa_type.value if hasattr(qa.qa_type, "value") else str(qa.qa_type),
        evidence=ev_list,
        verification=verif_dict
    )

