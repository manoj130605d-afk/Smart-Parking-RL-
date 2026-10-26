"""Train, evaluate, save the learned Q-table, and generate reproducible results."""
import csv, json, pickle, statistics
from pathlib import Path
from rl_agent import train_agent, evaluate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

def mean_rows(rows):
    keys = [k for k in rows[0] if k not in ("episode", "policy")]
    return {k: statistics.mean(row[k] for row in rows) for k in keys}

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    agent, rewards = train_agent(episodes=1800, seed=7)
    with open(ROOT / "trained_q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.q), f)
    with open(RESULTS / "training_rewards.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["episode", "total_reward"])
        w.writerows(enumerate(rewards, start=1))

    policies = {
        "Q-learning": ("rl", agent),
        "First available": ("first_available", None),
        "Nearest available": ("nearest_available", None),
        "Least occupied": ("least_occupied", None),
    }
    summary = []
    all_rows = []
    for name, (policy, trained_agent) in policies.items():
        rows = evaluate(agent=trained_agent, policy=policy, episodes=150, seed=2000)
        for row in rows:
            row["policy"] = name
            all_rows.append(row)
        m = mean_rows(rows)
        m["policy"] = name
        summary.append(m)
    write_csv(RESULTS / "evaluation_episodes.csv", all_rows)
    write_csv(RESULTS / "summary.csv", summary)

    try:
        import matplotlib.pyplot as plt
        # Smoothed training curve
        window = 60
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i-window+1)
            smoothed.append(sum(rewards[start:i+1]) / (i-start+1))
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(rewards)+1), smoothed)
        plt.xlabel("Training episode"); plt.ylabel("Mean reward (60-episode window)")
        plt.title("Q-learning training progress"); plt.tight_layout()
        plt.savefig(FIGURES / "training_curve.png", dpi=160); plt.close()

        metrics = ["avg_search_time", "avg_walking_distance", "acceptance_rate"]
        labels = ["Average search time (sim. units)", "Average walking distance (units)", "Acceptance rate"]
        for metric, label in zip(metrics, labels):
            plt.figure(figsize=(7, 4))
            plt.bar([x["policy"] for x in summary], [x[metric] for x in summary])
            plt.ylabel(label); plt.title("Policy comparison: " + label)
            plt.xticks(rotation=18, ha="right"); plt.tight_layout()
            plt.savefig(FIGURES / (metric + ".png"), dpi=160); plt.close()
    except ImportError:
        print("matplotlib not installed; CSV results were still generated.")

    print("Evaluation summary (means across 150 episodes per policy):")
    for row in summary:
        print(row)
    print("Saved Q-table, CSV results, and figures.")

if __name__ == "__main__":
    main()
