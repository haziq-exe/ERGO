import json
import sqlite3
import pytest
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.dataset import Code, GSM8K, Database, DataToText, Actions, Dataset
from evaluation.evaluator import CodeEvaluator, GSM8KEvaluator, DatabaseEvaluator, DataToTextEvaluator, ActionsEvaluator


def test_load_tiny_model():
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    assert tokenizer is not None
    assert model is not None


def test_evaluators_pass():
    sample_path = Path("sample_dataset/sample_dataset.json")

    # Actions task
    act_ds = Actions(sample_path)
    evaluator = ActionsEvaluator()
    model_answer = act_ds.data[0]["sim_model_ans"]
    res = evaluator.evaluate(act_ds, model_answer, 0)
    assert res["score"] == 1.0

    # CODE task
    code_ds = Code(sample_path)
    evaluator = CodeEvaluator()
    model_answer = "def add(a, b):\n    return a + b"
    res = evaluator.evaluate(code_ds, model_answer, 0)
    assert res["score"] == 1.0

    # GSM8K task["gsm8k"]
    gsm_ds = GSM8K(sample_path)
    evaluator = GSM8KEvaluator()
    model_answer = gsm_ds.data[0]["answer"]
    res = evaluator.evaluate(gsm_ds, model_answer, 0)
    assert res["score"] == 1.0

    # Data2Text task
    d2t_ds = DataToText(sample_path)
    evaluator = DataToTextEvaluator()
    model_answer = d2t_ds.data[0]["references"][0]
    res = evaluator.evaluate(d2t_ds, model_answer, 0)
    assert res["score"] == 1.0

    # Database task
    db_ds = Database(sample_path)
    evaluator = DatabaseEvaluator()
    model_answer = f"```sql\n{db_ds.data[0]['reference_sql']}\n```"
    db_path = Path("tests/resources/spider_db")
    res = evaluator.evaluate(db_ds, model_answer, db_path, 0)
    assert res["score"] == 1.0