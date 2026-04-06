import dspy
from pydantic import BaseModel


class HallucinationInput(BaseModel):
    context: str
    question: str
    answer: str


class HallucinationOutput(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str]
    reason: str


class HallucinationSignature(dspy.Signature):
    """Verify whether an answer is grounded in the provided context.
    An answer is grounded if every factual claim it makes can be supported
    by information explicitly present in the context. Flag any claim that
    is not directly supported. Minor paraphrasing and summarization are allowed.
    General knowledge statements not in the context count as unsupported."""
    input: HallucinationInput = dspy.InputField(desc="The retrieved context, the question, and the generated answer")
    output: HallucinationOutput = dspy.OutputField(desc="Whether the answer is grounded, any unsupported claims, and reasoning")


class HallucinationGuardrail(dspy.Module):
    def __init__(self):
        self.check = dspy.Predict(HallucinationSignature)

    def forward(self, context: str, question: str, answer: str) -> dspy.Prediction:
        return self.check(input=HallucinationInput(context=context, question=question, answer=answer))
