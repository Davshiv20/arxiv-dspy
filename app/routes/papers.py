from fastapi import APIRouter, HTTPException, Request
from app.schemas import (
    LoadRequest, LoadResponse,
    SummarizeResponse,
    QARequest, QAResponse,
    PapersListResponse, PaperInfo,
)
from app.services.store import add_paper, retrieve_chunks, has_paper, get_paper_text, get_paper, list_papers, make_paper_id
from app.services.ingestion import load_paper, chunk_text
from app.services.rag import RAG, Summarizer
from app.services.guardrails import RelevanceGuardrail, InjectionGuardrail, HallucinationGuardrail
from app.config import langfuse

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=PapersListResponse)
def get_papers():
    papers = list_papers()
    return PapersListResponse(papers=[PaperInfo(**p) for p in papers])


@router.post("/load", response_model=LoadResponse)
def load(req: LoadRequest, request: Request):
    with langfuse.start_as_current_observation(as_type="span", name="load-paper") as span:
        paper_id = make_paper_id(req.pdf_url)

        if has_paper(paper_id):
            span.update(output={"paper_id": paper_id, "already_existed": True})
            return LoadResponse(paper_id=paper_id, num_chunks=0, title=req.title, already_existed=True)

        with langfuse.start_as_current_observation(as_type="span", name="download-pdf") as dl_span:
            text = load_paper(req.pdf_url)
            dl_span.update(output=f"Extracted {len(text)} chars")

        with langfuse.start_as_current_observation(as_type="span", name="chunk-and-embed") as chunk_span:
            chunks = chunk_text(text)
            embedder = request.app.state.embedder
            num = add_paper(embedder, paper_id, req.pdf_url, req.title, text, chunks)
            chunk_span.update(output=f"{num} chunks stored")

        span.update(output={"paper_id": paper_id, "num_chunks": num})
    langfuse.flush()
    return LoadResponse(paper_id=paper_id, num_chunks=num, title=req.title)


@router.post("/{paper_id}/summarize", response_model=SummarizeResponse)
def summarize(paper_id: str):
    with langfuse.start_as_current_observation(as_type="span", name="summarize-paper") as span:
        text = get_paper_text(paper_id)
        if text is None:
            raise HTTPException(404, "Paper not found")

        with langfuse.start_as_current_observation(as_type="generation", name="dspy-summarize", model="gemini-2.5-flash") as gen:
            summarizer = Summarizer()
            result = summarizer(document=text)
            gen.update(
                input={"document_length": len(text)},
                output=result.output.summary,
            )

        span.update(output={"paper_id": paper_id, "summary_length": len(result.output.summary)})
    langfuse.flush()
    return SummarizeResponse(paper_id=paper_id, summary=result.output.summary)


@router.post("/{paper_id}/ask", response_model=QAResponse)
def ask(paper_id: str, req: QARequest, request: Request):
    with langfuse.start_as_current_observation(as_type="span", name="qa-paper") as span:
        paper = get_paper(paper_id)
        if paper is None:
            raise HTTPException(404, "Paper not found")

        # 1. Prompt injection guardrail (input)
        with langfuse.start_as_current_observation(as_type="generation", name="injection-guardrail", model="gemini-2.5-flash") as guard:
            injection_check = InjectionGuardrail()(question=req.question)
            guard.update(
                input={"question": req.question},
                output={"is_injection": injection_check.output.is_injection, "reason": injection_check.output.reason},
            )

        if injection_check.output.is_injection:
            langfuse.flush()
            raise HTTPException(
                status_code=400,
                detail=f"Prompt injection detected. Reason: {injection_check.output.reason}",
            )

        # 2. Relevance guardrail (input)
        with langfuse.start_as_current_observation(as_type="generation", name="relevance-guardrail", model="gemini-2.5-flash") as guard:
            relevance_check = RelevanceGuardrail()(paper_title=paper.get("title") or "Untitled", question=req.question)
            guard.update(
                input={"paper_title": paper.get("title"), "question": req.question},
                output={"is_relevant": relevance_check.output.is_relevant, "reason": relevance_check.output.reason},
            )

        if not relevance_check.output.is_relevant:
            langfuse.flush()
            raise HTTPException(
                status_code=400,
                detail=f"Question is not relevant to this paper. Reason: {relevance_check.output.reason}",
            )

        embedder = request.app.state.embedder

        with langfuse.start_as_current_observation(as_type="span", name="retrieve-chunks") as ret_span:
            chunks = retrieve_chunks(embedder, paper_id, req.question)
            ret_span.update(output=f"Retrieved {len(chunks)} chunks")

        with langfuse.start_as_current_observation(as_type="generation", name="dspy-qa", model="gemini-2.5-flash") as gen:
            rag = RAG(embedder)
            result = rag(question=req.question, paper_id=paper_id)
            gen.update(
                input={"question": req.question, "num_chunks": len(chunks)},
                output=result.output.answer,
            )

        # 3. Hallucination guardrail (output)
        context_str = "\n\n".join(chunks)
        with langfuse.start_as_current_observation(as_type="generation", name="hallucination-guardrail", model="gemini-2.5-flash") as guard:
            hallucination_check = HallucinationGuardrail()(
                context=context_str,
                question=req.question,
                answer=result.output.answer,
            )
            guard.update(
                input={"question": req.question, "answer_length": len(result.output.answer)},
                output={
                    "is_grounded": hallucination_check.output.is_grounded,
                    "unsupported_claims": hallucination_check.output.unsupported_claims,
                    "reason": hallucination_check.output.reason,
                },
            )

        if not hallucination_check.output.is_grounded:
            langfuse.flush()
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Generated answer is not grounded in the retrieved context.",
                    "unsupported_claims": hallucination_check.output.unsupported_claims,
                    "reason": hallucination_check.output.reason,
                },
            )

        span.update(output={"paper_id": paper_id, "question": req.question})
    langfuse.flush()
    return QAResponse(
        paper_id=paper_id,
        question=req.question,
        answer=result.output.answer,
        context=chunks,
    )
