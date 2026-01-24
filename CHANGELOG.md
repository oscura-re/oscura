# Changelog

All notable changes to Oscura will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Infrastructure** (.github/config/): Version-controlled GitHub configuration with main-branch-ruleset-template.json for replicable repository setup, main-branch-ruleset.json for current live config reference, comprehensive README.md documenting ruleset details and troubleshooting
- **Infrastructure** (.github/INFRASTRUCTURE.md): Complete IaC guide covering configuration philosophy, file structure, fork setup, merge queue strategy, best practices, and migration from old branch protection
- **Infrastructure** (.github/scripts/export-github-config.sh): Automated backup script for exporting live repository configuration (rulesets, settings, environments, labels, topics) to JSON files

### Changed

- **Documentation** (README.md): Accurate repositioning emphasizing workflow automation value, adds "Built On" transparency section showing integration with sigrok/scipy/ChipWhisperer, clarifies unique contributions (hypothesis-driven RE, DBC generation, Wireshark dissector automation) vs integrated capabilities (protocol decoding, signal processing), provides guidance on when to use Oscura vs other tools, maintains honest positioning as workflow automation platform that chains established tools
- **Infrastructure** (.github/scripts/setup-github-repo.sh): Idempotent repository setup script using rulesets API instead of deprecated branch protection, creates/updates ruleset from template, handles existing rulesets gracefully
- **Infrastructure** (.github/MERGE_QUEUE_SETUP.md): Updated to use repository rulesets instead of branch protection API, documents ALLGREEN strategy benefits, explains why explicit required_status_checks cause merge queue to get stuck, adds troubleshooting section for AWAITING_CHECKS issue

### Fixed

- **Infrastructure** (Repository Ruleset #12055878): Removed required_status_checks rule that caused merge queue to get stuck in AWAITING_CHECKS state (checks only ran on pull_request events, not merge_group events), now relies on ALLGREEN strategy which works for both event types

## [0.5.1] - 2026-01-24

**Clean History Release**: Production-ready framework with ultra-clean git history.

### Added

- Comprehensive hardware reverse engineering framework
- 112 working demonstrations across 19 categories
- 16+ protocol decoders (UART, SPI, I2C, CAN, LIN, FlexRay, JTAG, SWD, etc.)
- IEEE-compliant measurements (181/1241/1459/2414 standards)
- Side-channel analysis (DPA, CPA, timing attacks)
- Unknown protocol reverse engineering capabilities
- State machine extraction and field inference
- CRC/checksum recovery tools
- Diff coverage test suite achieving 80%+ coverage
- Python 3.13 test isolation fixes
- CI/CD pipeline with merge queue
- 5/5 quality validators passing

### Infrastructure

- Complete test coverage with parallel execution
- Automated quality checks (ruff, mypy, pytest)
- Claude Code orchestration with 6 specialized agents
- Comprehensive documentation and user guides
- Demonstration system with validation framework

## [0.1.2] - 2025-01-18

**Initial Public Release**: Foundation of the Oscura framework.

### Added

- Core signal processing and analysis capabilities
- Basic protocol decoding infrastructure
- Initial test suite and CI/CD pipeline
- Fundamental documentation and examples
- Package structure and build system
