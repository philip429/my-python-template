# Data science project template

A work-in-progress [Copier](https://copier.readthedocs.io/) template for data
exploration, analysis, and modeling with Python. It uses
[uv](https://docs.astral.sh/uv/), a `src` package layout, JupyterLab, Ruff, and pytest.

The template provides a working environment, exploration helpers, missing-data
summaries and imputation, prediction comparison plots, and regression tests.
The command-line pipeline is a starter: it creates output folders and logs their
location. Data ingestion, feature engineering, model training, and deployment
remain project-specific work. See the review and next steps below.

## Create a project

Install Git and uv, then run:

```bash
uvx copier copy https://github.com/philip429/my-python-template.git my-new-project --trust
cd my-new-project
uv run pytest
uv run jupyter lab
```

`--trust` permits the template's tasks: `git init` on creation and `uv sync` on
creation or update. `uv sync` installs the package in editable mode along with
its dependencies and development tools. Network access is needed on a cold cache.
To render without executing tasks, use `--skip-tasks` instead of `--trust`, then
run `git init` and `uv sync` yourself in the new directory.

| Question | Default | Meaning |
| --- | --- | --- |
| `project_name` | `my_data_project` | Lowercase letters, digits, underscores; starts with a letter, ends with a letter or digit; no Python keywords or reserved Windows filenames |
| `project_description` | `Data exploration and modeling project` | Project description |
| `author_name` | `Data Science Team` | Author or team |
| `python_version` | `3.12` | Python minor version, such as `3.12` or `3.13` |

The environment is restricted to the chosen Python minor version and records it
in `.python-version`. Python 3.12 and 3.13 are the CI targets. Other versions of
3.12 or newer pass input validation, but their dependencies must also support
that interpreter. The directory name and package name can differ; for example,
`my-new-project` can contain the package `customer_churn`.

For a non-interactive project with custom answers:

```bash
uvx copier copy https://github.com/philip429/my-python-template.git customer-churn --defaults --trust -d project_name=customer_churn -d python_version=3.12
```

Copier normally selects the latest release tag. Use `--vcs-ref=HEAD` to try the
latest repository changes. Commit the generated `.copier-answers.yml`,
`.python-version`, and `uv.lock` to preserve the template origin and environment.

## Update an existing project

Commit your work first. From the generated project directory:

```bash
uvx copier update --trust
uv run ruff check .
uv run pytest
```

Review the diff and resolve any Copier conflicts before committing the update.
To follow unreleased changes, use `uvx copier update --vcs-ref=HEAD --trust`.
Keep `.copier-answers.yml`: it records the source and answers used for updates.

## Work on this template

Edit `.jinja` files for content containing Copier variables. Plain `.py` files
are copied unchanged; use relative imports within reusable modules so they work
with every generated package name. The root README documents this repository;
`README.md.jinja` becomes the generated project's README. Copier prefers the
`.jinja` sibling when both have the same destination name.

The `notebooks/aux_notebooks/` folder contains unfinished reference notebooks.
It is kept in this repository and excluded from generated projects. These
notebooks are not part of the supported, reproducible starter workflow.

Run template rendering and input-validation tests from this directory:

```bash
uv run --with-requirements requirements-dev.txt pytest template_tests -q
```

Then generate and test a disposable project. `.cache/` is ignored by Git and
excluded from Copier output:

```bash
uv run --with-requirements requirements-dev.txt copier copy --defaults --skip-tasks --vcs-ref HEAD -d project_name=template_smoke . .cache/generated
cd .cache/generated
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run template-smoke
uv build
```

Use a fresh destination for a clean test; `--overwrite` can refresh an existing
disposable render, but does not remove stale files. Run checks against rendered
code, since Python tools cannot parse Jinja imports. `template_tests/` validates
rendering, input validation, metadata escaping, notebook structure, and excluded
files; generated `tests/` validates the reusable helpers. GitHub Actions runs
these checks on Windows and Linux with Python 3.12 and 3.13.

Before publishing a template release, review the generated diff and tag the
tested commit with a version such as `v1.2.0`. No license has been selected yet;
choose one before describing this template as licensed open-source software.

## Review and next steps

The structure is a useful foundation for an evolving data-science template.
This review corrected input validation and TOML escaping, missing-value
percentages, categorical KNN imputation, hard-coded model-helper imports and
column names, plotting callbacks, and paths for non-editable installations.
Existing misspelled helper names remain as compatibility aliases.

Dependency checks exposed two compatibility problems. UpSetPlot 0.9 fails under
pandas 3's copy-on-write behavior, so pandas is constrained below 3. PyJanitor
also needs a compatible pandas-flavor API: the tested combination is PyJanitor
0.31.0 with pandas-flavor 0.7. Revisit these constraints together when upgrading;
the generated tests exercise the affected APIs. UpSetPlot still emits upstream
deprecation warnings under pandas 2, but its plotting test passes.

Suggested next steps:

1. Add one reproducible example with a small synthetic or distributable dataset,
   ingestion, a train/test split, preprocessing, a baseline model, and held-out
   metrics. Wire `configs/params.yaml` into this example explicitly.
2. Use fitted scikit-learn pipelines for training and inference. The current KNN
   helper fits on every call, uses unscaled distances, and is for exploration.
3. Curate the reference notebooks individually: document data provenance,
   declare dependencies, replace Colab paths, clear outputs, and execute each
   from a clean environment before including it in generated projects.
4. As the API grows, consider a planned migration from `dataExploration` to
   separate exploration, preprocessing, and evaluation modules while preserving
   existing imports.
5. Choose a license and publish tested version tags.

The reference notebooks remain working notes and are not executed by the test
suite. Helper tests establish software behavior; they do not establish the
statistical validity of an analysis or imputation method for a given dataset.
