# install.py
import subprocess
import argparse

# Define all available groups and their descriptions for users
GROUPS_INFO = {
    "core": "Core functionality required by other modules.",
    "inference": "Inference module for running ML models. Automatically includes 'inference_server' and 'core'.",
    "inference_server": "Server wrapper for inference; automatically included with 'inference'.",
    "server": "Main server and leader/orchestrator functionality.",
    "ingest": "Data ingestion module; automatically includes 'core'.",
    "analytics": "Analytics tools for processed data.",
    "dev": "Developer tools: linting, testing, and debugging utilities."
}

# Define automatic group dependencies
AUTO_EXTRA_GROUPS = {
    "inference": ["inference_server", "core"],
    "ingest": ["core"]
}


def expand_groups(requested_groups):
    """Expand requested groups with their automatic extras recursively."""
    expanded = set(requested_groups)
    added = True
    while added:
        added = False
        for g in list(expanded):
            extras = AUTO_EXTRA_GROUPS.get(g, [])
            for e in extras:
                if e not in expanded:
                    expanded.add(e)
                    added = True
    return list(expanded)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Install selected poetry groups with automatic dependencies.\n\n"
            "User guidance:\n"
            "- Install 'inference' for inference tasks.\n"
            "- Install 'server' for the main server / orchestrator.\n"
            "- Install 'dev' if development tools are needed."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "groups",
        nargs="+",
        choices=GROUPS_INFO.keys(),
        help="One or more groups to install. Available groups:\n" +
             "\n".join(f"{g}: {d}" for g, d in GROUPS_INFO.items())
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which groups would be installed without running poetry."
    )

    args = parser.parse_args()

    # Expand groups with automatic extras
    all_groups = expand_groups(args.groups)
    print(f"Selected groups: {', '.join(args.groups)}")
    print(f"Installing groups (expanded with dependencies): {', '.join(all_groups)}")

    if args.dry_run:
        print("Dry run enabled; skipping poetry installation.")
        return

    # Run poetry install
    poetry_cmd = ["poetry", "install", "--no-root", "--with", ",".join(all_groups)]

    print(f"Running: {' '.join(poetry_cmd)}")
    subprocess.run(poetry_cmd, shell=True)

    # After running poetry install
    if "dev" in all_groups:
        print("Installing pre-commit hooks because 'dev' group is selected...")
        try:
            subprocess.run("poetry run pre-commit install", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install pre-commit hooks: {e}")
            print("Run manually: poetry run pre-commit install")


if __name__ == "__main__":
    main()
