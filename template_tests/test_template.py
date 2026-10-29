"""Check the rendered project, rather than treating Jinja files as Python."""

import ast
import json
import shutil
import tomllib
from pathlib import Path

import nbformat
import pytest
import yaml
from copier import run_copy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def template_source(tmp_path_factory):
    # Copy the working tree, including new files, without Git history or caches.
    destination = tmp_path_factory.mktemp("template") / "source"
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".cache",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "aux_notebooks",
        ),
    )
    return destination


@pytest.mark.parametrize("name", ["my_data_project", "customer_churn"])
def test_rendered_project(template_source, tmp_path, name):
    description = 'Analysis of "quoted" paths: C:\\data\nWith Unicode: café 🐍'
    author = 'A "quoted" author 🐍'
    output = tmp_path / "project"
    run_copy(
        str(template_source),
        output,
        defaults=True,
        skip_tasks=True,
        quiet=True,
        data={
            "project_name": name,
            "project_description": description,
            "author_name": author,
        },
    )
    config = tomllib.loads((output / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["description"] == description
    assert config["project"]["authors"][0]["name"] == author
    assert config["project"]["name"] == name.replace("_", "-")
    assert config["project"]["scripts"][name.replace("_", "-")] == f"{name}.main:main"
    assert config["project"]["requires-python"] == ">=3.12, <3.13"
    assert (output / ".python-version").read_text().strip() == "3.12"
    assert (output / "README.md").read_text(encoding="utf-8").startswith(f"# {name}\n")
    assert "data/**" in (output / ".gitignore").read_text()
    answers = yaml.safe_load(
        (output / ".copier-answers.yml").read_text(encoding="utf-8")
    )
    assert answers["project_name"] == name
    for path in output.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in output.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    notebook = nbformat.read(output / "notebooks/01-exploration.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == []
    assert not list(output.rglob("*.jinja"))
    for excluded in [
        "copier.yml",
        "requirements-dev.txt",
        "template_tests",
        ".github",
        "docs",
        ".cache",
        "notebooks/aux_notebooks",
    ]:
        assert not (output / excluded).exists()


@pytest.mark.parametrize(
    "name", ["class", "import", "async", "MyProject", "a-b", "../x", "trailing_", "con"]
)
def test_invalid_package_names(template_source, tmp_path, name):
    with pytest.raises(ValueError, match="package name|keyword"):
        run_copy(
            str(template_source),
            tmp_path / "project",
            defaults=True,
            skip_tasks=True,
            quiet=True,
            data={"project_name": name},
        )


@pytest.mark.parametrize("version", ["3", "3.12.1", "2.7", "3.9", "latest", "3.12junk"])
def test_invalid_python_versions(template_source, tmp_path, version):
    with pytest.raises(ValueError, match="Python minor version"):
        run_copy(
            str(template_source),
            tmp_path / "project",
            defaults=True,
            skip_tasks=True,
            quiet=True,
            data={"python_version": version},
        )
