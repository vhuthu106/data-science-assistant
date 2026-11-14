"""Planner: turns a question about a dataset into a short, validated list of steps.

A plan looks like:
    {
      "intent":   "average_age_by_class",
      "question": "average age by passenger class",
      "steps": [
        {"action": "group_analysis", "args": {"query": "Compute the mean Age for each Pclass"}}
      ]
    }
Each step's "query" is a specific instruction for that step, which the Worker turns into code.
"""
import json
import logging

from core.llm import chat

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {
    "data_analysis":        "summary statistics, counts, describing the data",
    "data_cleaning":        "missing values, data types, duplicates",
    "group_analysis":       "aggregates per category (mean, count, sum, ...)",
    "correlation_analysis": "relationships between numeric columns",
    "statistical_test":     "hypothesis tests (t-test, chi-square, ...)",
    "create_plot":          "one chart",
}
MAX_STEPS = 3


class PlanError(ValueError):
    """The model's reply was not a usable plan."""


def describe_dataset(df) -> str:
    """Column names and types only. Rows are never sent to the model."""
    if df is None:
        return "No dataset details available."
    cols = "\n".join(f"- {name}: {dtype}" for name, dtype in df.dtypes.astype(str).items())
    return f"{df.shape[0]} rows x {df.shape[1]} columns\nColumns (name: type):\n{cols}"


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply, even if it is wrapped in ``` fences or prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise PlanError("reply contains no JSON object")
    return json.loads(text[start:end + 1])


def validate_plan(raw, question: str) -> dict:
    """Check the plan's shape and return a clean copy. Raises PlanError if unusable."""
    if not isinstance(raw, dict):
        raise PlanError("plan is not a JSON object")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanError("plan has no steps")
    if len(steps) > MAX_STEPS:
        logger.warning("Plan had %d steps; keeping the first %d", len(steps), MAX_STEPS)

    clean_steps = []
    for step in steps[:MAX_STEPS]:
        if not isinstance(step, dict):
            raise PlanError("a step is not an object")
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            raise PlanError(f"unknown action {action!r}")
        args = step.get("args")
        query = args.get("query") if isinstance(args, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise PlanError(f"step {action!r} has no instruction")
        clean_steps.append({"action": action, "args": {"query": query.strip()}})

    return {"intent": str(raw.get("intent") or "analysis"), "question": question, "steps": clean_steps}


class Planner:
    def plan(self, question: str, df=None) -> dict:
        """Ask the LLM for a plan. If the reply is unusable, fall back to a rule-based plan.

        Errors talking to the provider (timeouts, no credit, overloaded) are NOT caught here:
        a fallback plan cannot help if the model is unreachable, so the caller shows the error.
        """
        actions = "\n".join(f"- {name}: {meaning}" for name, meaning in ALLOWED_ACTIONS.items())
        prompt = f"""Plan how to answer a question about a pandas DataFrame.

Dataset:
{describe_dataset(df)}

Question: "{question}"

Break the question into 1-{MAX_STEPS} steps. Each step uses ONE action from this list:
{actions}

Each step's "query" must say specifically what that step should do, using only columns that exist.
Reply with ONLY a JSON object in exactly this format:
{{"intent": "short_snake_case_summary",
 "steps": [{{"action": "group_analysis", "args": {{"query": "Compute the mean Age for each Pclass"}}}}]}}"""

        reply = chat([
            {"role": "system", "content": "You plan data analysis tasks. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": prompt},
        ])
        try:
            return validate_plan(extract_json(reply), question)
        except (json.JSONDecodeError, PlanError) as e:
            logger.warning("Unusable plan from the model (%s). Using the rule-based plan.", e)
            plan = self.rule_based_plan(question)
            plan["fallback_reason"] = str(e)
            return plan

    @staticmethod
    def rule_based_plan(question: str) -> dict:
        """Keyword-based single-step plan, used when the model's plan is unusable."""
        q = question.lower()
        if any(w in q for w in ["plot", "chart", "graph", "visualize", "histogram"]):
            action = "create_plot"
        elif any(w in q for w in ["correlation", "relationship"]):
            action = "correlation_analysis"
        elif any(w in q for w in ["clean", "missing", "null"]):
            action = "data_cleaning"
        elif any(w in q for w in ["average", "mean", "group by", " per ", " by "]):
            action = "group_analysis"
        else:
            action = "data_analysis"
        return {"intent": action, "question": question,
                "steps": [{"action": action, "args": {"query": question}}]}
