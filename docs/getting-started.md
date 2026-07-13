# Getting started

This guide assumes no Python project experience. Commands beginning with `$`
are typed into a terminal; do not type the `$` itself.

## 1. Check Python

```bash
python --version
```

You need Python 3.10, 3.11, 3.12, or 3.13. On some Linux and macOS systems the
command is named `python3`.

## 2. Download and isolate Word Forge

```bash
git clone https://github.com/Ace1928/word_forge.git
cd word_forge
python -m venv .venv
```

A virtual environment keeps Word Forge's libraries separate from other Python
applications.

Activate it:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## 3. Install

```bash
python -m pip install --upgrade pip
python -m pip install .
word-forge doctor
```

`doctor` labels required checks as `OK` or `ERROR`. Missing optional model
features are informational and do not prevent normal lexicon use.

## 4. Create and use a lexicon

```bash
word-forge init
word-forge add lexicon --definition "A collection of words and their meanings."
word-forge show lexicon
word-forge find collection
```

The default database belongs to your user account. To create a portable file in
the current directory, add `--db-path` to each command:

```bash
word-forge init --db-path ./team-terms.sqlite
word-forge add ontology --db-path ./team-terms.sqlite
word-forge stats --db-path ./team-terms.sqlite
```

## 5. Add another language

```bash
word-forge add libro --language es --definition "Conjunto de hojas encuadernadas."
word-forge translate book libro --from en --to es
word-forge show libro --language es
```

Language values are short standard tags such as `en`, `es`, `de`, `ar`,
`zh-Hans`, or `pt-BR`.

## 6. Enable algorithmic enrichment

Word Forge can use WordNet and sentiment data provided by NLTK:

```bash
python -m pip install ".[nlp]"
word-forge setup-nltk
```

This is the only step in the basic workflow that downloads lexical datasets.

## 7. Enable neural features only if wanted

```bash
python -m pip install ".[vector]"
word-forge start language --minutes 1 --with-vector
```

The first neural run may download a model and can require several gigabytes.
Core commands never do this implicitly.

## Common problems

- `word-forge: command not found`: reactivate `.venv`, or run
  `python -m word_forge` instead.
- `NLTK resource not found`: run `word-forge setup-nltk`.
- Permission error for data: set `WORD_FORGE_HOME` to a writable directory.
- Optional module missing: run `word-forge doctor` for the exact install command.
