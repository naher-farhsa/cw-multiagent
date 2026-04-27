from src.config.settings import validate_keys, RECURSION_LIMIT
from src.graph.workflow import build_workflow


def run():
    validate_keys()

    print("\n===== CONTENT WRITING MULTI-AGENT =====")
    topic = input("Enter the topic you want an article on: ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    user_request = (
        f"Research the topic '{topic}' thoroughly and write a well-structured "
        f"Markdown article on it. Save the final article to a .md file."
    )

    app = build_workflow()

    print("\n----- WORKFLOW STARTED -----\n")
    for step in app.stream(
        {"messages": [("user", user_request)]},
        {"recursion_limit": RECURSION_LIMIT},
    ):
        for node, payload in step.items():
            print(f"[{node}] -> {payload}")
            print("---")

    print("\n===== WORKFLOW FINISHED =====")


if __name__ == "__main__":
    run()
