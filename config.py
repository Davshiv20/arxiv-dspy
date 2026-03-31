

import dspy
import os

OR_API_KEY = os.getenv("OPENROUTER_API_KEY")


lm = dspy.LM("openrouter/stepfun/step-3.5-flash:free", api_key = OR_API_KEY )

dspy.configure(lm=lm)