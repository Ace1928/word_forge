# Fish completion for the Word Forge command line.
# Install: copy this file to ~/.config/fish/completions/word-forge.fish

function __word_forge_needs_command
    not __fish_seen_subcommand_from init add show lookup find search relate translate sense delete-sense stats integrity backup restore doctor gui serve start graph vector emotion demo setup-nltk
end

complete -c word-forge -n __word_forge_needs_command -s h -l help -d 'Show help'
complete -c word-forge -n __word_forge_needs_command -s V -l version -d 'Show version'
complete -c word-forge -n __word_forge_needs_command -s q -l quiet -d 'Suppress non-error output'
complete -c word-forge -n __word_forge_needs_command -s v -l verbose -d 'Enable diagnostic output'
complete -c word-forge -n __word_forge_needs_command -s c -l config -r -d 'Load JSON or TOML configuration'

complete -c word-forge -n __word_forge_needs_command -a init -d 'Create the database schema'
complete -c word-forge -n __word_forge_needs_command -a add -d 'Add or update an entry'
complete -c word-forge -n __word_forge_needs_command -a show -d 'Show an exact entry'
complete -c word-forge -n __word_forge_needs_command -a lookup -d 'Alias for show'
complete -c word-forge -n __word_forge_needs_command -a find -d 'Search terms and definitions'
complete -c word-forge -n __word_forge_needs_command -a search -d 'Alias for find'
complete -c word-forge -n __word_forge_needs_command -a relate -d 'Create a typed relationship'
complete -c word-forge -n __word_forge_needs_command -a translate -d 'Create a translation'
complete -c word-forge -n __word_forge_needs_command -a sense -d 'Add or revise a meaning'
complete -c word-forge -n __word_forge_needs_command -a delete-sense -d 'Tombstone a meaning'
complete -c word-forge -n __word_forge_needs_command -a stats -d 'Show record counts'
complete -c word-forge -n __word_forge_needs_command -a integrity -d 'Check database health'
complete -c word-forge -n __word_forge_needs_command -a backup -d 'Create a database snapshot'
complete -c word-forge -n __word_forge_needs_command -a restore -d 'Restore a database snapshot'
complete -c word-forge -n __word_forge_needs_command -a doctor -d 'Check local capabilities'
complete -c word-forge -n __word_forge_needs_command -a gui -d 'Open the secure local workspace'
complete -c word-forge -n __word_forge_needs_command -a serve -d 'Alias for gui'
complete -c word-forge -n __word_forge_needs_command -a start -d 'Run optional enrichment workers'
complete -c word-forge -n __word_forge_needs_command -a graph -d 'Manage semantic graphs'
complete -c word-forge -n __word_forge_needs_command -a vector -d 'Manage optional neural indexes'
complete -c word-forge -n __word_forge_needs_command -a emotion -d 'Run emotion analysis'
complete -c word-forge -n __word_forge_needs_command -a demo -d 'Run a prepared demonstration'
complete -c word-forge -n __word_forge_needs_command -a setup-nltk -d 'Download optional NLTK datasets'
