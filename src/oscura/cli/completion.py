"""Shell completion support for Oscura CLI.

Generates completion scripts for bash, zsh, and fish shells.


Example:
    $ oscura --install-completion bash
    $ oscura --show-completion bash > ~/.bash_completion.d/oscura
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_completion_script(shell: str) -> str:
    """Get completion script for specified shell.

    Args:
        shell: Shell type ('bash', 'zsh', or 'fish').

    Returns:
        Completion script content.

    Raises:
        ValueError: If shell type is unsupported.
    """
    if shell == "bash":
        return _get_bash_completion()
    elif shell == "zsh":
        return _get_zsh_completion()
    elif shell == "fish":
        return _get_fish_completion()
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def install_completion(shell: str) -> Path:
    """Install completion script for specified shell.

    Args:
        shell: Shell type ('bash', 'zsh', or 'fish').

    Returns:
        Path where completion was installed.

    Raises:
        ValueError: If shell type is unsupported.
    """
    script = get_completion_script(shell)
    home = Path.home()

    if shell == "bash":
        completion_dir = home / ".bash_completion.d"
        completion_dir.mkdir(exist_ok=True)
        completion_file = completion_dir / "oscura"
    elif shell == "zsh":
        completion_dir = home / ".zsh" / "completion"
        completion_dir.mkdir(parents=True, exist_ok=True)
        completion_file = completion_dir / "_oscura"
    elif shell == "fish":
        completion_dir = home / ".config" / "fish" / "completions"
        completion_dir.mkdir(parents=True, exist_ok=True)
        completion_file = completion_dir / "oscura.fish"
    else:
        raise ValueError(f"Unsupported shell: {shell}")

    with open(completion_file, "w") as f:
        f.write(script)

    return completion_file


def _get_bash_completion() -> str:
    """Get bash completion script."""
    return """# Bash completion for oscura

_oscura_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="analyze decode export visualize benchmark validate config plugins characterize batch compare shell tutorial"

    # Global options
    local global_opts="--help --version --verbose --quiet --config --json"

    # If we're on the first argument, complete commands or global options
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- ${cur}) )
        return 0
    fi

    # Get the command
    local cmd="${COMP_WORDS[1]}"

    # Command-specific completions
    case "${cmd}" in
        analyze|decode|export|visualize)
            # Complete file paths and help
            if [[ ${cur} == -* ]]; then
                COMPREPLY=( $(compgen -W "--help" -- ${cur}) )
            else
                COMPREPLY=( $(compgen -f -X '!*.@(wfm|vcd|csv|pcap|wav)' -- ${cur}) )
            fi
            ;;
        config)
            local config_opts="--show --set --edit --init --path --help"
            COMPREPLY=( $(compgen -W "${config_opts}" -- ${cur}) )
            ;;
        plugins)
            local plugin_cmds="list info install remove update"
            if [[ ${COMP_CWORD} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "${plugin_cmds}" -- ${cur}) )
            fi
            ;;
        *)
            # Default to --help for other commands
            if [[ ${cur} == -* ]]; then
                COMPREPLY=( $(compgen -W "--help" -- ${cur}) )
            fi
            ;;
    esac
}

complete -F _oscura_completion oscura
"""


def _get_zsh_completion() -> str:
    """Get zsh completion script."""
    return """#compdef oscura

_oscura() {
    local -a commands
    commands=(
        'analyze:Run full analysis workflow'
        'decode:Decode protocol data'
        'export:Export analysis results'
        'visualize:Launch interactive viewer'
        'benchmark:Run performance benchmarks'
        'validate:Validate protocol specification'
        'config:Manage configuration'
        'plugins:Manage plugins'
        'characterize:Characterize signal'
        'batch:Batch process files'
        'compare:Compare signals'
        'shell:Start interactive shell'
        'tutorial:Run interactive tutorial'
    )

    local -a file_args
    file_args=(
        '*:waveform file:_files -g "*.{wfm,vcd,csv,pcap,wav}"'
    )

    _arguments -C \\
        '(-h --help)'{-h,--help}'[Show help message]' \\
        '(-v --verbose)'{-v,--verbose}'[Increase verbosity]' \\
        '--config[Configuration file]:config file:_files' \\
        '(-q --quiet)'{-q,--quiet}'[Quiet mode]' \\
        '--json[JSON output mode]' \\
        '1: :->command' \\
        '*:: :->args'

    case $state in
        command)
            _describe -t commands 'oscura commands' commands
            ;;
        args)
            case $words[1] in
                analyze|decode|visualize)
                    _files -g "*.{wfm,vcd,csv,pcap,wav}"
                    ;;
                config)
                    _arguments \\
                        '--show[Show configuration]' \\
                        '--set[Set value]:key=value:' \\
                        '--edit[Edit configuration]' \\
                        '--init[Initialize configuration]' \\
                        '--path[Show config path]'
                    ;;
            esac
            ;;
    esac
}

_oscura
"""


def _get_fish_completion() -> str:
    """Get fish completion script."""
    return """# Fish completion for oscura

# Main commands
complete -c oscura -n "__fish_use_subcommand" -a analyze -d "Run full analysis workflow"
complete -c oscura -n "__fish_use_subcommand" -a decode -d "Decode protocol data"
complete -c oscura -n "__fish_use_subcommand" -a export -d "Export analysis results"
complete -c oscura -n "__fish_use_subcommand" -a visualize -d "Launch interactive viewer"
complete -c oscura -n "__fish_use_subcommand" -a benchmark -d "Run performance benchmarks"
complete -c oscura -n "__fish_use_subcommand" -a validate -d "Validate protocol specification"
complete -c oscura -n "__fish_use_subcommand" -a config -d "Manage configuration"
complete -c oscura -n "__fish_use_subcommand" -a plugins -d "Manage plugins"
complete -c oscura -n "__fish_use_subcommand" -a characterize -d "Characterize signal"
complete -c oscura -n "__fish_use_subcommand" -a batch -d "Batch process files"
complete -c oscura -n "__fish_use_subcommand" -a compare -d "Compare signals"
complete -c oscura -n "__fish_use_subcommand" -a shell -d "Start interactive shell"
complete -c oscura -n "__fish_use_subcommand" -a tutorial -d "Run interactive tutorial"

# Global options
complete -c oscura -s h -l help -d "Show help message"
complete -c oscura -s v -l verbose -d "Increase verbosity"
complete -c oscura -s q -l quiet -d "Quiet mode"
complete -c oscura -l config -d "Configuration file" -r
complete -c oscura -l json -d "JSON output mode"

# analyze subcommand
complete -c oscura -n "__fish_seen_subcommand_from analyze" -l protocol -d "Protocol hint"
complete -c oscura -n "__fish_seen_subcommand_from analyze" -l export-dir -d "Export directory" -r
complete -c oscura -n "__fish_seen_subcommand_from analyze" -s i -l interactive -d "Interactive mode"
complete -c oscura -n "__fish_seen_subcommand_from analyze" -l output -d "Output format" -a "json csv html table"

# decode subcommand
complete -c oscura -n "__fish_seen_subcommand_from decode" -l protocol -d "Protocol type" -a "uart spi i2c can auto"
complete -c oscura -n "__fish_seen_subcommand_from decode" -l baud-rate -d "Baud rate" -r
complete -c oscura -n "__fish_seen_subcommand_from decode" -l show-errors -d "Show only errors"

# config subcommand
complete -c oscura -n "__fish_seen_subcommand_from config" -l show -d "Show configuration"
complete -c oscura -n "__fish_seen_subcommand_from config" -l set -d "Set value" -r
complete -c oscura -n "__fish_seen_subcommand_from config" -l edit -d "Edit configuration"
complete -c oscura -n "__fish_seen_subcommand_from config" -l init -d "Initialize configuration"
complete -c oscura -n "__fish_seen_subcommand_from config" -l path -d "Show config path"

# File completions for commands that take file arguments
for cmd in analyze decode visualize export
    complete -c oscura -n "__fish_seen_subcommand_from $cmd" -F -a '(__fish_complete_suffix .wfm .vcd .csv .pcap .wav)'
end
"""


if __name__ == "__main__":
    # Allow running as: python -m oscura.cli.completion bash
    if len(sys.argv) > 1:
        shell_type = sys.argv[1]
        print(get_completion_script(shell_type))
