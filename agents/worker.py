"""Worker: turns each plan step into pandas code, runs it, and collects the results."""
import logging
import re

from core.entropy_logger import EntropyLogger
from core.executor import new_namespace, run_code, snapshot, user_variables
from core.llm import chat
from agents.planner import Planner, describe_dataset

logger = logging.getLogger(__name__)

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """Return the code from a reply: the first ``` block if there is one, else the whole reply."""
    match = CODE_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip()


class Worker:
    def __init__(self):
        self.logger = EntropyLogger()   # ✅ database logger
        self.planner = Planner()        # ✅ feedback planner

    def handle(self, plan, dataset_manager, max_retries=2):
        df = dataset_manager.get_dataset()
        if df is None:
            return {"error": "No dataset loaded", "steps": []}

        question = plan.get("question", "")
        namespace = new_namespace(df)          # shared by ALL steps of this plan
        outputs = {"steps": []}

        for step in plan["steps"]:
            action = step["action"]
            instruction = step.get("args", {}).get("query", "")
            retries = 0

            while True:
                code = self._generate_code(question, action, instruction, namespace)
                trial = snapshot(namespace)
                result = run_code(code, trial)

                if result.success:
                    namespace.clear()
                    namespace.update(trial)    # keep this step's changes for later steps
                    break

                retries += 1
                if retries > max_retries:
                    break
                # 🔄 Ask planner to revise plan  (replaced in Step 6)
                revised_plan = self._feedback_to_planner(plan, action, result.error)
                if revised_plan:
                    step = revised_plan["steps"][0]
                    action = step["action"]
                    instruction = step.get("args", {}).get("query", "")

            outputs["steps"].append({"action": action, "instruction": instruction, **vars(result)})

        return outputs

    def _feedback_to_planner(self, plan, action, error_msg):
        """Ask planner to fix the plan if worker execution failed"""
        try:
            revised = self.planner.plan(f"Revise due to error: {error_msg}")
            return revised
        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            return None

    def _generate_code(self, question, action, instruction, namespace):
        """Ask the LLM for code that performs one step."""
        earlier = user_variables(namespace)
        earlier_text = ", ".join(f"{n} ({t})" for n, t in earlier.items()) or "none"
        prompt = f"""Write Python code for one step of a data analysis.

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
3. Do not read or write files and do not use the network.
4. Never use inplace=True; assign instead, e.g. df["Age"] = df["Age"].fillna(df["Age"].median())
5. Use only columns that exist.

Reply with only the code."""
        reply = chat([
            {"role": "system", "content": "You write short, correct pandas code. Reply with code only."},
            {"role": "user", "content": prompt},
        ])
        return extract_code(reply)
