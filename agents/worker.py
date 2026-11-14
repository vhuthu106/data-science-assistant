import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tempfile
import json
import logging
import re  # Added for regex pattern matching
import warnings  # Added for warning handling

# Import entropy logger + planner for feedback loop
from core.entropy_logger import EntropyLogger
from core.llm import chat
from agents.planner import Planner

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self):
        self.logger = EntropyLogger()   # ✅ database logger
        self.planner = Planner()        # ✅ feedback planner

    def handle(self, plan, dataset_manager, max_retries=2):
        outputs = {"images": [], "tables": [], "results": {}}
        df = dataset_manager.get_dataset()
        if df is None:
            return {"error": "No dataset loaded"}

        user_query = plan.get("intent", "unknown_task")

        for step in plan["steps"]:
            action = step["action"]
            args = step.get("args", {})
            retries = 0

            while retries <= max_retries:
                try:
                    # --- Generate code
                    generated_code = self._generate_code(action, args, df)

                    # --- Execute
                    result = self._execute_code(generated_code, df)

                    if result.get("success"):
                        outputs["results"][action] = result
                        if result.get("image"):
                            outputs["images"].append(result["image"])
                        if result.get("table") is not None:
                            outputs["tables"].append(result["table"])
                        
                        # ✅ Log success
                        #self.logger.log_interaction(user_query, f"{action} success", error_count=retries)
                        break

                    else:
                        raise Exception(result.get("error", "Unknown error"))

                except Exception as e:
                    retries += 1
                    error_msg = str(e)

                    # ❌ Log failure
                    #self.logger.log_interaction(user_query, f"{action} failed: {error_msg}", error_count=retries)

                    if retries <= max_retries:
                        # 🔄 Ask planner to revise plan
                        revised_plan = self._feedback_to_planner(plan, action, error_msg)
                        if revised_plan:
                            plan = revised_plan
                            step = plan["steps"][0]  # take revised first step
                            action, args = step["action"], step.get("args", {})
                            continue
                    else:
                        # 🛑 Final fallback
                        fallback_plan = self.planner.rule_based_plan(user_query)
                        outputs["results"][action] = {
                            "success": False,
                            "error": error_msg,
                            "plan_used": "rule_based_fallback",
                            "fallback_plan": fallback_plan
                        }
                        break

        return outputs
    
    def _feedback_to_planner(self, plan, action, error_msg):
        """Ask planner to fix the plan if worker execution failed"""
        try:
            revised = self.planner.plan(f"Revise due to error: {error_msg}")
            return revised
        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            return None    
    
    def _generate_code(self, action, args, df):
        """Generate code with the LLM"""
        prompt = self._create_code_prompt(action, args, df)

        code = chat([
            {"role": "system", "content": "You are a helpful code generation assistant. Return only executable Python code. Avoid pandas inplace operations with chained assignment as they cause warnings."},
            {"role": "user", "content": prompt}
        ]).strip()
        code = self._clean_code(code)
        code = self._validate_and_fix_code(code, df)
        return code

    def _create_code_prompt(self, action, args, df):
        query = args.get('query', '')
        return f"""
        Generate Python code to perform: {action}
        Query: {query}
        
        Dataframe info:
        - Name: {getattr(df, 'name', 'Unknown')}
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {df.dtypes.to_dict()}
        
        **IMPORTANT REQUIREMENTS:**
        1. Return ONLY Python code without any explanations
        2. Use modern pandas practices (avoid inplace=True with chained assignment)
        3. Never use: df['col'].method(inplace=True) - this causes warnings
        4. Instead use: df = df.method() or df['col'] = df['col'].method()
        5. Handle missing values appropriately with .copy() when needed
        6. Include necessary imports
        7. Make sure the code is complete and executable
        
        **BAD EXAMPLE (causes warnings):**
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        
        **GOOD EXAMPLE:**
        df = df.copy()
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        
        Example for data cleaning:
        import pandas as pd
        import numpy as np
        
        # Create a copy to avoid warnings
        df_clean = df.copy()
        
        # Handle missing values properly
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
        df_clean['Embarked'] = df_clean['Embarked'].fillna('Unknown')
        
        # Return cleaned dataframe
        result = df_clean
        """
    
    def _validate_and_fix_code(self, code, df):
        """Validate generated code and fix common pandas issues"""
        # List of problematic patterns and their fixes
        patterns_and_fixes = [
            # Fix inplace operations with chained assignment
            (r"df\[['\"](.*?)['\"]\]\.(.*?)\(.*?inplace\s*=\s*True", r"df = df.copy()\ndf['\1'] = df['\1'].\2("),
            (r"df\.(.*?)\(.*?inplace\s*=\s*True", r"df = df.\1("),
            
            # Ensure copy is used when modifying
            (r"^(?!.*\.copy\(\))(.*df\[.*\] =.*)", r"df = df.copy()\n\1"),
            
            # Add missing imports
            (r"^import pandas as pd", r"import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns"),
        ]
        
        # Apply fixes
        for pattern, fix in patterns_and_fixes:
            code = re.sub(pattern, fix, code, flags=re.MULTILINE)
        
        # Ensure copy is used if dataframe is modified
        if any(mod_pattern in code for mod_pattern in ["df[", "df.", "= df", "df ="]):
            if "df.copy()" not in code and "df = df.copy()" not in code:
                code = "df = df.copy()\n" + code
        
        return code
    
    def _clean_code(self, code):
        """Clean generated code"""
        if not code:
            return "print('No code generated')"
            
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        # Remove any non-code content
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            if not line.strip().startswith(('//', '/*', '*/', '#')) or line.strip().startswith('# '):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _execute_code(self, code, df):
        try:
            df_copy = df.copy()
            
            # Create a function to capture print output
            print_output = []
            def custom_print(*args, **kwargs):
                output = " ".join(str(arg) for arg in args)
                print_output.append(output)
            
            # Capture warnings too
            warning_messages = []
            
            def custom_warning(message, category, filename, lineno, file=None, line=None):
                warning_messages.append(f"{category.__name__}: {message}")
            
            # Set up execution environment
            local_vars = {
                'df': df_copy,
                'pd': pd,
                'plt': plt,
                'np': np,
                'sns': sns,
                'print': custom_print,
                'warnings': warnings
            }
            
            # Execute with warnings captured
            with warnings.catch_warnings(record=True) as w:
                warnings.showwarning = custom_warning
                exec(code, {}, local_vars)
                
                # Capture any warnings
                for warning in w:
                    warning_messages.append(f"{warning.category.__name__}: {warning.message}")
            
            result = {
                "success": True, 
                "code": code,
                "print_output": print_output,
                "warnings": warning_messages  # Include warnings in result
            }
            
            # Check for plots
            if plt.gcf().get_axes():
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
                    plt.close()
                    result["image"] = tmp.name
            
            # Check for any returned results
            for var_name in local_vars:
                if var_name not in ['df', 'pd', 'plt', 'np', 'sns', 'print', 'custom_print', 'warnings'] and not var_name.startswith('_'):
                    var_value = local_vars[var_name]
                    if isinstance(var_value, (pd.DataFrame, pd.Series)):
                        result["table"] = var_value
                    elif isinstance(var_value, (str, int, float, list, dict)):
                        result["output"] = var_value
            
            return result
            
        except Exception as e:
            return {"error": str(e), "code": code, "success": False}