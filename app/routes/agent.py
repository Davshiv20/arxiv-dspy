from fastapi import APIRouter
from app.schemas import AgentRequest, AgentResponse
from app.services.agent import ResearchAgent
from app.config import langfuse

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    with langfuse.start_as_current_observation(as_type="span", name="research-agent") as span:
        with langfuse.start_as_current_observation(as_type="generation", name="dspy-react", model="gemini-2.5-flash") as gen:
            agent = ResearchAgent()
            result = agent(question=req.question)
            gen.update(
                input={"question": req.question},
                output=result.output.answer,
            )
        span.update(output={"question": req.question, "answer": result.output.answer})
    langfuse.flush()
    return AgentResponse(
        question=req.question,
        answer=result.output.answer,
        reasoning=result.output.reasoning,
    )
