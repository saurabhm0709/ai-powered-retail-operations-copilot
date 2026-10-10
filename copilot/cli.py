"""Command line entry point: python -m copilot.cli "text" [--persona Cashier] [--store ST003]."""

import argparse

from copilot import orchestrator


# Parses arguments, runs the pipeline once and prints the CopilotResponse as JSON.
def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Operations Copilot (one request)")
    parser.add_argument("query", help="the employee's request text")
    parser.add_argument("--persona", default=None, help="e.g. Cashier, Store Manager")
    parser.add_argument("--store", default=None, help="store id from the UI, e.g. ST003")
    args = parser.parse_args()
    response = orchestrator.run(args.query, persona=args.persona, store_hint=args.store)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
