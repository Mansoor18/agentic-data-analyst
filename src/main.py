from agent import analyse
import os

if __name__ == "__main__":
    # Use our churn dataset — we already know it well
    csv_path = os.path.expanduser(
        "~/Desktop/Projects/DS-AI-ML/churn-prediction/data/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    questions = [
        "What are the top 3 factors driving customer churn? Provide statistical evidence.",
        "What is the average monthly charge for churned vs non-churned customers?",
    ]

    for question in questions:
        analyse(csv_path, question)
        print("\n" + "="*60 + "\n")