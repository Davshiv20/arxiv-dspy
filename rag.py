import dspy
from retriever import retrieve


class RAG(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        chunks = retrieve(question)
        context = "\n\n".join(chunks)
        return self.generate_answer(context=context, question=question)


class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought("document -> summary")

    def forward(self, document: str) -> dspy.Prediction:
        # Truncate to avoid token limits
        return self.summarize(document=document[:4000])
