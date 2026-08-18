import ast
from pathlib import Path


def test_hybrid_reader_uses_the_template_row_count_for_final_row_handling():
    """KCET's final rows are 60/120/180/240, not NEET's 45-row cadence."""
    module = ast.parse(Path("ml_omr/hybrid_reader.py").read_text(encoding="utf-8"))
    scan_answers_ml = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "scan_answers_ml"
    )

    assert "questions_per_column" in [arg.arg for arg in scan_answers_ml.args.args]

    modulo_checks = [
        node
        for node in ast.walk(scan_answers_ml)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and isinstance(node.right, ast.Name)
        and node.right.id == "questions_per_column"
    ]
    assert modulo_checks

    scanner_source = Path("scanner.py").read_text(encoding="utf-8")
    assert "questions_per_column=int(" in scanner_source
