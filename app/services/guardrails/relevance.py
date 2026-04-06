import dspy
from pydantic import BaseModel


class RelevanceInput(BaseModel):
    paper_title: str
    question: str


class RelevanceOutput(BaseModel):
    is_relevant: bool
    reason: str


class RelevanceSignature(dspy.Signature):
    """Decide whether the user's question is relevant to the given research paper.
    A question is relevant if answering it would require information from the paper,
    or if it concerns the paper's topic, methods, results, or contributions.
    Reject questions that are off-topic, general knowledge, or unrelated to the paper."""
    input: RelevanceInput = dspy.InputField(desc="The paper title and user question")
    output: RelevanceOutput = dspy.OutputField(desc="Relevance decision with reason")


class RelevanceGuardrail(dspy.Module):
    def __init__(self):
        self.check = dspy.Predict(RelevanceSignature)

    def forward(self, paper_title: str, question: str) -> dspy.Prediction:
        return self.check(input=RelevanceInput(paper_title=paper_title, question=question))
