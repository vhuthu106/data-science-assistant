import json
import logging

from core.llm import chat

logger = logging.getLogger(__name__)

class Planner:
    def plan(self, user_query: str):
        return self._request_plan(user_query)

    def _request_plan(self, user_query: str):
        """Use the LLM for planning"""
        prompt = f"""
You are a planner agent for data science tasks. Convert the user request into a valid JSON plan with EXACTLY this structure:
{{
  "intent": "brief_description_of_the_task",
  "steps": [
    {{"action": "specific_action_name", "args": {{"query": "user_query_here"}}}}
  ],
  "analysis": ["steps for data prep or stats"],
  "visualization": ["steps for plots"],
  "validation": ["checks for correctness/errors"],
  "tools": ["pandas", "matplotlib", "seaborn"]
}}

Available actions: create_plot, data_analysis, data_cleaning, correlation_analysis, group_analysis, statistical_test

User request: "{user_query}"

Return ONLY the JSON object without any additional text, explanations, or code blocks.
"""
        
        try:
            content = chat(
                [
                    {"role": "system", "content": "You are a planning assistant that returns only valid JSON. No additional text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
            )
            content = self._clean_json_response(content.strip())

            if not content or content.isspace():
                raise Exception("Empty response from OpenAI")

            plan = json.loads(content)

            # ✅ Ensure steps exist
            if "steps" not in plan or not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
                inferred_action = "data_analysis"
                q = user_query.lower()
                if any(x in q for x in ["plot", "chart", "visualize"]):
                    inferred_action = "create_plot"
                elif any(x in q for x in ["group", "average", "mean"]):
                    inferred_action = "group_analysis"
                elif any(x in q for x in ["correlation", "relationship"]):
                    inferred_action = "correlation_analysis"
                elif any(x in q for x in ["clean", "missing", "null"]):
                    inferred_action = "data_cleaning"

                plan["steps"] = [{"action": inferred_action, "args": {"query": user_query}}]
                logger.info(f"✅ Auto-added steps to plan: {plan['steps']}")

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Content: {content if 'content' in locals() else 'None'}")
            return self._create_fallback_plan(user_query)
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def _clean_json_response(self, content):
        """Clean the response to extract pure JSON"""
        if not content:
            return "{}"
        
        # Remove markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Remove any non-JSON content before the first {
        if '{' in content:
            content = content[content.find('{'):]
        
        # Remove any non-JSON content after the last }
        if '}' in content:
            content = content[:content.rfind('}') + 1]
        
        return content
    
    def _create_fallback_plan(self, user_query: str):
        """Rule-based fallback planning"""
        user_query_lower = user_query.lower()
        
        if any(word in user_query_lower for word in ['plot', 'chart', 'graph', 'visualize']):
            return {
                "intent": "data_visualization",
                "steps": [{"action": "create_plot", "args": {"query": user_query}}],
                "tools": ["matplotlib", "seaborn"]
            }
        elif any(word in user_query_lower for word in ['analyze', 'statistics', 'describe', 'summary']):
            return {
                "intent": "data_analysis",
                "steps": [{"action": "data_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "numpy"]
            }
        elif any(word in user_query_lower for word in ['clean', 'process', 'transform', 'filter']):
            return {
                "intent": "data_processing",
                "steps": [{"action": "data_cleaning", "args": {"query": user_query}}],
                "tools": ["pandas"]
            }
        elif any(word in user_query_lower for word in ['correlation', 'relationship']):
            return {
                "intent": "correlation_analysis",
                "steps": [{"action": "correlation_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "seaborn"]
            }
        elif any(word in user_query_lower for word in ['average', 'mean', 'group by']):
            return {
                "intent": "group_analysis",
                "steps": [{"action": "group_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "matplotlib"]
            }
        else:
            return {
                "intent": "general_analysis",
                "steps": [{"action": "data_analysis", "args": {"query": user_query}}],
                "tools": ["pandas", "matplotlib"]
            }
    
    def critique(self, plan: dict):
        """Ask the model to review its own plan and suggest improvements"""
        critique_prompt = f"""
You are a senior data science reviewer. Review the following plan for errors, missing steps, or unclear reasoning.
Suggest improvements without changing the intent. Return JSON like:
{{
  "issues": ["list of problems"],
  "improvements": ["suggested fixes"],
  "confidence": "high/medium/low"
}}

Plan: {json.dumps(plan, indent=2)}
"""
        try:
            return self._request_plan(critique_prompt)
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return {"issues": ["Critique unavailable"], "improvements": [], "confidence": "low"}
