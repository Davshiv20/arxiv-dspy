import dspy
from sentence_transformers import SentenceTransformer
from app.services.store import retrieve_chunks
from app.schemas import QAInput, QAOutput, SummaryInput, SummaryOutput


class QASignature(dspy.Signature):
    """Answer questions about a research paper based on the provided context."""
    input: QAInput = dspy.InputField(desc="The context and question to answer")
    output: QAOutput = dspy.OutputField(desc="The answer with confidence level")


class SummarySignature(dspy.Signature):
    """Summarize an academic paper."""
    input: SummaryInput = dspy.InputField(desc="The paper text to summarize")
    output: SummaryOutput = dspy.OutputField(desc="Summary with key findings")


class RAG(dspy.Module):
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.generate_answer = dspy.ChainOfThought(QASignature)

    def forward(self, question: str, paper_id: str = "__default__") -> dspy.Prediction:
        chunks = retrieve_chunks(self.embedder, paper_id, question)
        context = "\n\n".join(chunks)
        return self.generate_answer(input=QAInput(context=context, question=question))


class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarySignature)

    def forward(self, document: str) -> dspy.Prediction:
        return self.summarize(input=SummaryInput(document=document[:4000]))


# 1. Chunking
# 2. Evals


# 3. Guardrails