import ast
from pathlib import Path

REPO_ROOT = Path('/home/runner/work/AIF-Gen/AIF-Gen')
TRAINER_PATH = REPO_ROOT / 'benchmarks/dpo/continual_dpo_trainer.py'
SCRIPT_PATH = REPO_ROOT / 'benchmarks/dpo/dpo_continual.py'


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def get_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f'Class {class_name} not found in {module}')


def get_function(module: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f'Function {function_name} not found in {module}')


def get_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f'Method {method_name} not found in {class_node.name}')


def called_attribute_names(node: ast.AST) -> list[str]:
    names = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            names.append(call.func.attr)
    return names


def test_trainer_no_longer_overrides_generic_log_or_accelerator_creation() -> None:
    trainer_module = parse_module(TRAINER_PATH)
    trainer_class = get_class(trainer_module, 'ContinualDPOTrainer')
    method_names = {
        node.name for node in trainer_class.body if isinstance(node, ast.FunctionDef)
    }

    assert 'log' not in method_names
    assert 'create_accelerator_and_postprocess' not in method_names
    assert 'set_task_datasets' in method_names
    assert 'generate_completions_table' in method_names



def test_generate_completions_uses_prompt_tokens_for_generation_samples() -> None:
    trainer_module = parse_module(TRAINER_PATH)
    trainer_class = get_class(trainer_module, 'ContinualDPOTrainer')
    method = get_method(trainer_class, 'generate_completions_table')

    subscripts = []
    for node in ast.walk(method):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == 'batch' and isinstance(node.slice, ast.Constant):
                subscripts.append(node.slice.value)

    assert 'prompt_input_ids' in subscripts



def test_dpo_script_uses_single_trainer_lifecycle_and_task_switch_method() -> None:
    script_module = parse_module(SCRIPT_PATH)
    main_function = get_function(script_module, 'main')

    trainer_inits = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'ContinualDPOTrainer'
    ]
    assert len(trainer_inits) == 1

    attr_calls = called_attribute_names(main_function)
    assert 'set_task_datasets' in attr_calls
    assert 'evaluate_policy' in attr_calls
    assert 'generate_completions_table' in attr_calls

    trainer_log_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'log'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'trainer'
    ]
    assert trainer_log_calls == []



def test_dpo_script_drops_duplicate_local_import_and_keeps_package_import() -> None:
    script_module = parse_module(SCRIPT_PATH)
    imported_modules = [
        node.module
        for node in script_module.body
        if isinstance(node, ast.ImportFrom)
    ]

    assert 'continual_dpo_trainer' not in imported_modules
    assert 'benchmarks.dpo.continual_dpo_trainer' in imported_modules



def test_dpo_script_has_explicit_reward_model_loader_helper() -> None:
    script_module = parse_module(SCRIPT_PATH)
    load_helper = get_function(script_module, 'load_reward_model_for_task')
    source = ast.unparse(load_helper)

    assert 'AutoModelForSequenceClassification.from_pretrained' in source
    assert 'torch_dtype' in source
