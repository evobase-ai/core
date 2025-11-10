# RUN_ME.py - One-click demo
from src.evobase import EvoBase

print("EvoBase Framework Demo - Nov 7, 2025")
agent = EvoBase(device="cpu")  # or "cuda"

# Simulate 3 days
for day in range(1, 4):
    print(f"\n--- Day {day} ---")
    agent.interact(f"Day {day}: I love learning!")
    agent.interact(f"Day {day}: Forget this junk.")
    agent.sleep_distill()
    agent.sqia_check()

agent.save("final_state.json")
print("\nDemo complete! Check final_state.json")
