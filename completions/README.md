# Shell completion

Word Forge includes completion definitions for Bash, Zsh, and Fish. These files
are optional; the CLI works without them.

## Bash

Copy `word-forge.bash` to a directory loaded by `bash-completion`, commonly:

```bash
sudo cp completions/word-forge.bash \
  /usr/share/bash-completion/completions/word-forge
```

## Zsh

Copy `_word-forge` into a directory on Zsh's `fpath`, then run `compinit`:

```zsh
mkdir -p ~/.zsh/completions
cp completions/_word-forge ~/.zsh/completions/
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

## Fish

```fish
mkdir -p ~/.config/fish/completions
cp completions/word-forge.fish ~/.config/fish/completions/
```
