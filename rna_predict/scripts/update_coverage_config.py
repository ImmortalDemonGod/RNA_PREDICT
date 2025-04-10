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

CONFIG_FILE = ".coverage_config.json"

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

def list_modules(config):
    """List all modules in the configuration."""
    print("\nCurrent Coverage Status:")
    print("=" * 50)
    print(f"Base Coverage: {config['base_coverage']}%")
    print(f"Current Coverage: {config['current_coverage']}%")
    print(f"Maximum Coverage: {config['max_coverage']}%")

    print("\nCurrent Module Categorization:")
    print("=" * 50)

    for category, modules in config["module_categories"].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for module in modules:
            print(f"  - {module}")

    print("\nCurrent Coverage Goals by Phase:")
    print("=" * 50)

    for phase, details in config["phase_coverage"].items():
        print(f"\n{phase.replace('_', ' ').title()} Phase ({details['start_date']} to {details['end_date']}):")
        for key, value in details.items():
            if key not in ["start_date", "end_date"]:
                print(f"  - {key.replace('_', ' ').title()}: {value}%")

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

def update_phase(config, phase_name, overall=None, critical=None, standard=None, utility=None):
    """Update coverage goals for a phase."""
    if phase_name not in config["phase_coverage"]:
        print(f"Error: Phase '{phase_name}' not found. Valid phases are: {', '.join(config['phase_coverage'].keys())}")
        sys.exit(1)

    if overall is not None:
        config["phase_coverage"][phase_name]["overall"] = overall
        print(f"Updated overall coverage goal for {phase_name} to {overall}%")

    if critical is not None:
        config["phase_coverage"][phase_name]["critical_modules"] = critical
        print(f"Updated critical modules coverage goal for {phase_name} to {critical}%")

    if standard is not None:
        config["phase_coverage"][phase_name]["standard_modules"] = standard
        print(f"Updated standard modules coverage goal for {phase_name} to {standard}%")

    if utility is not None:
        config["phase_coverage"][phase_name]["utility_modules"] = utility
        print(f"Updated utility modules coverage goal for {phase_name} to {utility}%")

    return config

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

    config = load_config()

    if args.list_modules:
        list_modules(config)
        return

    if args.add_module and args.category:
        config = add_module(config, args.add_module, args.category)
        save_config(config)
        return

    if args.update_phase:
        config = update_phase(config, args.update_phase, args.overall, args.critical, args.standard, args.utility)
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
