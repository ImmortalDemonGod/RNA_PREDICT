#!/usr/bin/env python3
"""
Update Coverage Configuration Tool

This script helps update the module categorization in the .coverage_config.json file.
It can be used to add new modules, change module categories, or update coverage goals.

Usage:
    python update_coverage_config.py --add-module <module_name> --category <category_name>
    python update_coverage_config.py --list-modules
    python update_coverage_config.py --update-phase <phase_name> --overall <percentage>
"""

import argparse
import json
import os
import sys
from datetime import datetime

DEFAULT_CONFIG_FILE = ".coverage_config.json"
CONFIG_FILE = DEFAULT_CONFIG_FILE  # Will be updated in main() if --config is provided

def load_config():
    """Load the coverage configuration file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found. Run run_failing_tests.sh first to generate it.")
        sys.exit(1)

    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save the coverage configuration file."""
    config["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {CONFIG_FILE}")

def _print_section_header(title):
    """Print a section header with consistent formatting."""
    print(f"\n{title}")
    print("=" * 50)

def _print_coverage_status(config):
    """Print the current coverage status."""
    _print_section_header("Current Coverage Status")
    print(f"Base Coverage: {config['base_coverage']}%")
    print(f"Current Coverage: {config['current_coverage']}%")
    print(f"Maximum Coverage: {config['max_coverage']}%")

def _print_module_categories(config):
    """Print the current module categorization."""
    _print_section_header("Current Module Categorization")

    for category, modules in config["module_categories"].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for module in modules:
            print(f"  - {module}")

def _print_phase_goals(config):
    """Print the coverage goals for each phase."""
    _print_section_header("Current Coverage Goals by Phase")

    for phase, details in config["phase_coverage"].items():
        print(f"\n{phase.replace('_', ' ').title()} Phase ({details['start_date']} to {details['end_date']}):")
        for key, value in details.items():
            if key not in ["start_date", "end_date"]:
                print(f"  - {key.replace('_', ' ').title()}: {value}%")

def list_modules(config):
    """List all modules in the configuration."""
    _print_coverage_status(config)
    _print_module_categories(config)
    _print_phase_goals(config)

def add_module(config, module_name, category):
    """Add a module to a category."""
    if category not in config["module_categories"]:
        print(f"Error: Category '{category}' not found. Valid categories are: {', '.join(config['module_categories'].keys())}")
        sys.exit(1)

    # Check if module already exists in any category
    for cat, modules in config["module_categories"].items():
        if module_name in modules:
            if cat == category:
                print(f"Module '{module_name}' is already in category '{category}'")
                return config
            else:
                print(f"Moving module '{module_name}' from '{cat}' to '{category}'")
                config["module_categories"][cat].remove(module_name)

    # Add module to the specified category
    config["module_categories"][category].append(module_name)
    print(f"Added module '{module_name}' to category '{category}'")

    return config

class CoverageGoals:
    """Class to hold coverage goal values."""
    def __init__(self, overall=None, critical=None, standard=None, utility=None):
        self.overall = overall
        self.critical = critical
        self.standard = standard
        self.utility = utility

def validate_coverage_value(name, value):
    """Validate that a coverage value is within the valid range."""
    if value is not None and not (0 <= value <= 100):
        print(f"Error: {name} coverage must be between 0 and 100, got {value}")
        sys.exit(1)
    return value

def update_phase(config, phase_name, goals=None, **kwargs):
    """Update coverage goals for a phase.

    Args:
        config: The configuration dictionary
        phase_name: The name of the phase to update
        goals: A CoverageGoals object containing the goals to update
        **kwargs: Alternative way to specify goals (overall, critical, standard, utility)
    """
    if phase_name not in config["phase_coverage"]:
        print(f"Error: Phase '{phase_name}' not found. Valid phases are: {', '.join(config['phase_coverage'].keys())}")
        sys.exit(1)

    # Create goals object if individual parameters were provided
    if goals is None:
        goals = CoverageGoals(
            overall=kwargs.get("overall"),
            critical=kwargs.get("critical"),
            standard=kwargs.get("standard"),
            utility=kwargs.get("utility")
        )

    # Update the configuration with validated values
    _update_phase_goal(config, phase_name, "overall", goals.overall)
    _update_phase_goal(config, phase_name, "critical_modules", goals.critical)
    _update_phase_goal(config, phase_name, "standard_modules", goals.standard)
    _update_phase_goal(config, phase_name, "utility_modules", goals.utility)

    return config

def _update_phase_goal(config, phase_name, goal_key, value):
    """Helper function to update a specific goal for a phase."""
    if value is not None:
        validate_coverage_value(goal_key.replace("_modules", ""), value)
        config["phase_coverage"][phase_name][goal_key] = value
        display_name = goal_key.replace("_modules", "").title()
        print(f"Updated {display_name} coverage goal for {phase_name} to {value}%")

def update_base_coverage(config, base_coverage=None, current_coverage=None):
    """Update base and current coverage values."""
    if base_coverage is not None:
        config["base_coverage"] = base_coverage
        print(f"Updated base coverage to {base_coverage}%")

    if current_coverage is not None:
        config["current_coverage"] = current_coverage
        print(f"Updated current coverage to {current_coverage}%")

    return config

def main():
    parser = argparse.ArgumentParser(description="Update coverage configuration")
    parser.add_argument("--config", default=DEFAULT_CONFIG_FILE,
                       help=f"Path to configuration file (default: {DEFAULT_CONFIG_FILE})")
    parser.add_argument("--list-modules", action="store_true", help="List all modules in the configuration")
    parser.add_argument("--add-module", help="Add a module to a category")
    parser.add_argument("--category", choices=["critical_modules", "standard_modules", "utility_modules"],
                        help="Category for the module")
    parser.add_argument("--update-phase", choices=["exploration", "development", "optimization", "final_submission"],
                        help="Update coverage goals for a phase")
    parser.add_argument("--overall", type=int, help="Overall coverage goal percentage")
    parser.add_argument("--critical", type=int, help="Critical modules coverage goal percentage")
    parser.add_argument("--standard", type=int, help="Standard modules coverage goal percentage")
    parser.add_argument("--utility", type=int, help="Utility modules coverage goal percentage")
    parser.add_argument("--base-coverage", type=int, help="Base coverage percentage")
    parser.add_argument("--current-coverage", type=int, help="Current coverage percentage")

    args = parser.parse_args()

    # Update global CONFIG_FILE if custom path provided
    global CONFIG_FILE
    CONFIG_FILE = args.config

    config = load_config()

    if args.list_modules:
        list_modules(config)
        return

    if args.add_module and args.category:
        config = add_module(config, args.add_module, args.category)
        save_config(config)
        return

    if args.update_phase:
        goals = CoverageGoals(
            overall=args.overall,
            critical=args.critical,
            standard=args.standard,
            utility=args.utility
        )
        config = update_phase(config, args.update_phase, goals=goals)
        save_config(config)
        return

    if args.base_coverage is not None or args.current_coverage is not None:
        config = update_base_coverage(config, args.base_coverage, args.current_coverage)
        save_config(config)
        return

    # If no action specified, show help
    parser.print_help()

if __name__ == "__main__":
    main()
