"""CLI entry point for ArguMind."""
import sys
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="ArguMind — Evidence Reasoning Agent")
    parser.add_argument("query", nargs="?", help="Research question")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter your research question: ").strip()
    if not query:
        print("No query provided.")
        sys.exit(1)

    print(f"\n🔍 ArguMind processing: {query}\n")

    from agents.orchestrator import ArguMindOrchestrator
    orchestrator = ArguMindOrchestrator()
    result = orchestrator.run(query)

    if args.json:
        print(json.dumps({
            "query": query,
            "answer": result.get("final_response"),
            "confidence": result.get("confidence"),
            "evidence_count": len(result.get("evidence_set", [])),
            "iteration_count": result.get("iteration_count"),
            "reasoning_trace": result.get("reasoning_trace"),
        }, indent=2))
    else:
        print("=" * 60)
        print(f"ANSWER (confidence: {result.get('confidence', 0):.0%})")
        print("=" * 60)
        print(result.get("final_response", "No response."))
        print()
        print("Reasoning trace:")
        for step in result.get("reasoning_trace", []):
            print(f"  → {step}")


if __name__ == "__main__":
    main()
