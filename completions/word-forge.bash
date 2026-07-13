# Bash completion for the Word Forge command line.
# Install: copy this file to /usr/share/bash-completion/completions/word-forge

_word_forge_completion() {
    local current
    current="${COMP_WORDS[COMP_CWORD]}"
    local commands="init add show lookup find search relate translate sense delete-sense stats integrity backup restore doctor gui serve start graph vector emotion demo setup-nltk"
    local options="--help --version --quiet --verbose --config"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands} ${options}" -- "${current}"))
        return
    fi

    case "${COMP_WORDS[1]}" in
        add)
            options="--definition --part-of-speech --language --example --source --base-term --if-revision --db-path --json --help"
            ;;
        show|lookup)
            options="--language --db-path --json --help"
            ;;
        find|search)
            options="--language --limit --db-path --json --help"
            ;;
        relate)
            options="--kind --db-path --json --help"
            ;;
        translate)
            options="--from --to --source --one-way --db-path --json --help"
            ;;
        sense)
            options="--language --part-of-speech --domain --register --source --confidence --review-state --sense-id --if-revision --db-path --json --help"
            ;;
        delete-sense)
            options="--if-revision --db-path --json --help"
            ;;
        stats|init)
            options="--db-path --json --help"
            ;;
        integrity)
            options="--db-path --json --help"
            ;;
        backup|restore)
            options="--db-path --force --json --help"
            ;;
        gui|serve)
            options="--db-path --host --port --no-open --help"
            ;;
        *)
            options="--help --db-path --json"
            ;;
    esac
    COMPREPLY=($(compgen -W "${options}" -- "${current}"))
}

complete -F _word_forge_completion word-forge word_forge
