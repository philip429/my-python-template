# Data science project template

A Copier template for reproducible Python data-science projects using `uv`, a
`src` package layout, JupyterLab, Ruff, and pytest.

## Create a project

```bash
uvx copier copy https://github.com/philip429/my-python-template.git my-new-project --trust
```

Copier initializes Git and runs `uv sync` after rendering. Review the generated
`README.md` for the project structure and common commands.

## Update an existing project

From the generated project directory:

```bash
uvx copier update --trust
```

Commit `.copier-answers.yml`; Copier uses it to track the template and answers
needed for future updates.
