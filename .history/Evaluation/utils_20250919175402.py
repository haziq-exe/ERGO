from eval_bfcl import ast_checker, ast_parse
import re
import sacrebleu
import sys
import tempfile
import subprocess
import sqlite3


class EvalUtils:
    def __init__(self):
        pass

class ActionsEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def extract_function_block(self, text):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        start = text.find('[')
        if start == -1:
            return ''

        level = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                level += 1
            elif text[i] == ']':
                level -= 1
                if level == 0:
                    block = text[start:i+1]
                    return self.clean_function_block(block)
        return ''

    def clean_function_block(self, block):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        block = block.replace('\n', '').replace('\r', '').replace('\t', '')
        block = ' '.join(block.split())

        # Remove "..." wrapping function calls only
        block = re.sub(r'"\s*([a-zA-Z_][a-zA-Z0-9_\.]*\s*\([^"]*\))\s*"', r'\1', block)

        # Remove space after [ and before ]
        block = re.sub(r'\[\s+', '[', block)
        block = re.sub(r'\s+\]', ']', block)

        return block


    def extract_all_function_blocks(self, text):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        """
        blocks = []
        start_positions = [i for i, c in enumerate(text) if c == '[']

        for start in start_positions:
            level = 0
            found = False
            for i in range(start, len(text)):
                if text[i] == '[':
                    level += 1
                elif text[i] == ']':
                    level -= 1
                    if level == 0:
                        block = text[start:i+1]
                        blocks.append(self.clean_function_block(block))
                        found = True
                        break
            if found:
                # Skip any nested [ inside this block — move to next outer [
                continue
        return blocks

    def evaluator_function(self, predicted_answer, sample):
        """
        FUNCTION TAKEN ENTIRELY FROM https://github.com/microsoft/lost_in_conversation -- https://arxiv.org/pdf/2505.06120
        Evaluate if the predicted function call matches the expected format and functionality.
        """

        try:
            decoded_output = ast_parse(predicted_answer.strip(), sample["language"])
        except Exception as e:
            # print(f"\033[94mPredicted answer:{predicted_answer}\033[0m")
            return {"is_correct": False, "error": "Failing to parse the predicted answer as an AST"}

        result = ast_checker(
            sample["function"],
            decoded_output,
            sample["reference_answer"],
            sample["language"],
            sample["test_category"],
            "gpt-4o"
        )
        score = 1 if result["valid"] else 0
        return {"is_correct": result["valid"], "score": score, "error": result["error"]}


class DataToTextEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def D2T_evaluator_function(self, extracted_answer, sample):
        # ToTTo has multiple references per example
        references = sample["references"]
        bleu = sacrebleu.corpus_bleu([extracted_answer.strip()], [[ref.strip()] for ref in references])
        return bleu.score / 100.0
    
class DatabaseEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()


    def extract_sql_query(self, text):
        # Match content inside ```sql ... ``` block
        match = re.search(r'```sql(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean leading/trailing whitespace
            return match.group(1).strip()
        else:
            return None


    def extract_sql_queries(self, text):
        """
        Extract all SQL queries from a text blob.
        Captures both fenced code blocks (```sql ... ```) and standalone statements ending with semicolons.
        """
        queries = []

        # 1) Fenced SQL blocks
        fenced = re.findall(r'```sql\s*(.*?)```', text, flags=re.IGNORECASE | re.DOTALL)
        queries.extend(q.strip() for q in fenced if q.strip())

        # 2) Standalone statements (SELECT/INSERT/UPDATE/DELETE/CREATE/ALTER/DROP) ending with ;
        #    Avoid re-capturing fenced blocks by skipping matches that span lines containing ```
        stmt_pattern = re.compile(
            r'(?:\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.+?;)',
            flags=re.IGNORECASE | re.DOTALL
        )
        for m in stmt_pattern.finditer(text):
            snippet = m.group().strip()
            # skip if it's identical to a fenced block
            if not any(snippet == f.strip() or snippet in f for f in fenced):
                queries.append(snippet)
        # if not queries:
        #   queries.append(text)
        return queries

    def run_query(db_path, sql):
      conn = sqlite3.connect(db_path)
      # decode any bytes with replacement on errors
      conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
      cur  = conn.cursor()
      try:
          cur.execute(sql)
          rows = cur.fetchall()
      finally:
          conn.close()
      return rows
    
class CodeEvalUtils(EvalUtils):
    def __init__(self):
        super().__init__()

    def extract_all_function_blocks_and_names(self, code):
        """
        Extracts all top-level Python function blocks (def ...) along with any import
        statements that appear immediately before each function. Returns a list of tuples:
        [(full_code_with_imports, function_name), ...].
        """
        lines = code.strip().splitlines()
        n = len(lines)
        results = []
        import_lines = []
        i = 0

        while i < n:
            line = lines[i]

            # If this line is an import, collect it and move on
            if re.match(r'^\s*import\s+\w', line) or re.match(r'^\s*from\s+\w+\s+import\s+', line):
                import_lines.append(line)
                i += 1
                continue

            # If this line is a top‐level function definition
            func_match = re.match(r'^(\s*)def\s+([A-Za-z_]\w*)\s*\(.*\)\s*:', line)
            if func_match:
                func_indent = len(func_match.group(1))
                func_name = func_match.group(2)

                # Collect the entire function block
                func_block = [lines[i]]
                j = i + 1
                while j < n:
                    next_line = lines[j]
                    # Blank lines inside the block are allowed
                    if next_line.strip() == "":
                        func_block.append(next_line)
                        j += 1
                        continue

                    # Check indentation: if indent > func_indent, it's still inside
                    indent_level = len(next_line) - len(next_line.lstrip())
                    if indent_level > func_indent:
                        func_block.append(next_line)
                        j += 1
                    else:
                        break

                # Combine the imports collected so far with this function block
                full_code = "\n".join(import_lines + [""] + func_block).rstrip()
                results.append((full_code, func_name))

                # Reset import_lines for the next function
                import_lines = []
                # Continue scanning from the line after this function block
                i = j
                continue

            # Neither an import nor a function definition: move on
            i += 1

        return results
    

    def extract_first_function_block_and_name(self, code):
        """
        Extracts the first top-level Python function (def ...) block and its name,
        along with any import statements above it.
        Returns the full function code (with imports) and function name.
        """
        lines = code.strip().splitlines()
        import_lines = []
        func_start_idx = None
        func_indent = None
        func_name = None

        # Collect top-level import statements before the function
        for i, line in enumerate(lines):
            if re.match(r'^\s*import\s+\w', line) or re.match(r'^\s*from\s+\w+\s+import\s+', line):
                import_lines.append(line)
            # elif re.match(r'^\s*def\s+[a-zA-Z_]\w*\s*\(.*\)\s*:', line):
            #     func_start_idx = i
            #     match = re.match(r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            #     func_indent = len(match.group(1))
            #     func_name = match.group(2)
            #     break
            elif re.match(r'^\s*def\s+[a-zA-Z_]\w*\s*\(.*\)\s*(->\s*[^\s:]+)?\s*:', line):
                func_start_idx = i
                match = re.match(r'^(\s*)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*\)\s*(->\s*[^\s:]+)?\s*:', line)
                func_indent = len(match.group(1))
                func_name = match.group(2)
                break

        if func_start_idx is None:
            return None, None

        # Collect lines in the function block
        func_lines = lines[func_start_idx:]
        collected = [func_lines[0]]

        for line in func_lines[1:]:
            if line.strip() == "":
                collected.append(line)
            elif len(line) - len(line.lstrip()) >= func_indent + 1:
                collected.append(line)
            else:
                break

        return "\n".join(import_lines + [""] + collected).rstrip(), func_name


    def run_function_and_check(func_name, user_code, test_cases):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            user_code_path = f.name
            f.write("import math\n")
            f.write(user_code)

        # print(user_code)
        runner_code = f"""
    import json
    import sys
    import ast
    import math
    import importlib.util

    spec = importlib.util.spec_from_file_location("tempmod", "{user_code_path}")
    tempmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tempmod)

    failures = []
    def parse_argument(line):
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            return line[1:-1]  # Strip outer double quotes, treat as string
        try:
            return ast.literal_eval(line)
        except:
            return line  # fallback

    for idx, case in enumerate({test_cases}):
        input_lines = case["input"].splitlines()
        args = [parse_argument(line) for line in input_lines]

        try:
            expected = ast.literal_eval(case["output"])
        except:
            expected = case["output"]

        if expected == "true" or expected == "Yes" or expected == "yes":
            expected = True

        if expected == "false" or expected == "No" or expected == "no":
            expected = False

        try:
            got = getattr(tempmod, "{func_name}")(*args)
        except Exception as e:
            failures.append(f"{{args}} raised {{e!r}}")
            continue

        if got == "Yes" or got == "yes":
        got = True
        if got == "No" or got == "no":
        got = False

        if got and got != expected:
            args = args.reverse()
            try:
            got = getattr(tempmod, "{func_name}")(*args)
            except Exception as e:
            failures.append(f"{{args}} raised {{e!r}}")
            continue

            if got == "Yes" or got == "yes":
            got = True
            if got == "No" or got == "no":
            got = False

            if got != expected:
            failures.append(f"Test {{idx}} :- {{args}}: got={{got!r}}, expected={{expected!r}}")

    if failures:
        print(json.dumps({{"ok": False, "errors": failures}}))
        sys.exit(1)
    else:
        print(json.dumps({{"ok": True}}))
        sys.exit(0)
    """

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            runner_path = f.name
            f.write(runner_code)

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=3
            )
        except:
            print("TIMEOUT ERROR")
            return False

        if result.returncode == 0:
            return True
        else:
            return False