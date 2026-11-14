"""Worker: turns each plan step into pandas code, runs it, and collects the results."""
import logging
import re

from core.executor import new_namespace, run_code, snapshot, user_variables
from core.llm import LLMUnavailable, chat
from core.safety import ALLOWED_MODULES
from agents.planner import describe_dataset

logger = logging.getLogger(__name__)

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
MAX_REPAIRS = 2   # after the first attempt, how many times the model may fix its own code


def extract_code(reply: str) -> str:
    """Return the code from a reply: the first ``` block if there is one, else the whole reply."""
    match = CODE_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip()


class Worker:
    def handle(self, plan, dataset_manager):
        """Run every step of the plan in order, sharing one namespace.

        Stops early (marking the remaining steps as skipped) if a step still fails after
        its repairs, or if the model provider can't be reached. Finished steps are kept.
        """
        df = dataset_manager.get_dataset()
        if df is None:
            return {"error": "No dataset loaded", "steps": []}

        question = plan.get("question", "")
        namespace = new_namespace(df)          # shared by ALL steps of this plan
        outputs = {"steps": [], "stopped": None}

        for index, step in enumerate(plan["steps"]):
            action = step["action"]
            instruction = step.get("args", {}).get("query", "")
            try:
                record = self._run_step(question, action, instruction, namespace)
            except LLMUnavailable as e:
                outputs["stopped"] = str(e)
                outputs["steps"].append(self._not_run(action, instruction, error=str(e)))
            else:
                outputs["steps"].append(record)
                if not record["success"]:
                    outputs["stopped"] = f"Step {index + 1} failed after {len(record['failed_attempts'])} attempts."

            if outputs["stopped"]:
                for later in plan["steps"][index + 1:]:
                    outputs["steps"].append(self._not_run(later["action"], later.get("args", {}).get("query", ""),
                                                          skipped=True))
                break

        return outputs

    def _run_step(self, question, action, instruction, namespace):
        """Generate code, run it, and if it fails let the model repair it using the error."""
        messages = [
            {"role": "system", "content": "You write short, correct pandas code. Reply with code only."},
            {"role": "user", "content": self._code_prompt(question, action, instruction, namespace)},
        ]
        failed_attempts = []
        for attempt in range(1 + MAX_REPAIRS):
            code = extract_code(chat(messages))
            trial = snapshot(namespace)
            result = run_code(code, trial)
            if result.success:
                namespace.clear()
                namespace.update(trial)        # keep this step's changes for later steps
                break
            failed_attempts.append({"code": code, "error": result.error})
            logger.info("Step %r attempt %d failed: %s", action, attempt + 1, result.error)
            # Show the model its own code and the exact error, and ask for a fix.
            printed = f"\nPrinted output before the error:\n{result.output[-1000:]}" if result.output else ""
            messages += [
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {"role": "user", "content": f"That code failed with this error:\n{result.error}{printed}\n\n"
                                            "Fix the problem. Reply with the complete corrected code only."},
            ]
        return {"action": action, "instruction": instruction, "failed_attempts": failed_attempts,
                "skipped": False, **vars(result)}

    @staticmethod
    def _not_run(action, instruction, error=None, skipped=False):
        """Record for a step that produced no code result (provider error, or skipped)."""
        return {"action": action, "instruction": instruction, "success": False, "skipped": skipped,
                "error": error, "code": "", "output": "", "warnings": [], "figures": [],
                "result": None, "failed_attempts": []}

    def _code_prompt(self, question, action, instruction, namespace):
        """The instructions for writing one step's code."""
        earlier = user_variables(namespace)
        earlier_text = ", ".join(f"{n} ({t})" for n, t in earlier.items()) or "none"
        return f"""Write Python code for one step of a data analysis.

Overall question: {question}
This step ({action}): {instruction}

Already available - do NOT re-create, re-import or reload them:
- df: the dataset as a pandas DataFrame. Changes you make to df are kept for later steps.
- pd, np, plt (matplotlib.pyplot), sns (seaborn)
- variables from earlier steps: {earlier_text}

Current dataset:
{describe_dataset(namespace["df"])}

Rules:
1. Put the main answer (a DataFrame, Series, number or short text) in a variable named `result`.
2. For charts, just draw them with plt or sns. Do NOT call plt.show(), plt.savefig() or plt.close().
3. Do not read or write files and do not use the network. You may only import: {", ".join(sorted(ALLOWED_MODULES))}.
4. Never use inplace=True; assign instead, e.g. df["Age"] = df["Age"].fillna(df["Age"].median())
5. Use only columns that exist.

Reply with only the code."""
