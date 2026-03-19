import json
import csv
import random

# All the building blocks
actions = ["turn", "continue", "arrive", "stop", "take_stairs", "take_lift", "warning"]
directions = ["left", "right"]
distances = ["5m", "10m", "15m", "20m", "30m", "50m"]
landmarks = ["escalator", "main entrance", "reception desk", "lift", "staircase",
             "platform 3", "exit", "information booth", "shop", "ticket gate"]
hazards = ["wet floor", "step down", "low ceiling", "construction barrier", "crowd ahead"]
floors = ["ground floor", "first floor", "second floor", "basement level"]

dataset = []

# --- Turn instructions ---
for direction in directions:
    for distance in distances:
        for landmark in landmarks[:5]:
            inp = f"action: turn direction: {direction} distance: {distance} landmark: {landmark}"
            # Write 2-3 variations of the target to add diversity
            targets = [
                f"In {distance}, turn {direction} towards the {landmark}.",
                f"Turn {direction} in {distance}, heading towards the {landmark}.",
                f"After {distance}, take a {direction} turn at the {landmark}.",
            ]
            for t in targets:
                dataset.append({"input": inp, "target": t})

# --- Continue instructions ---
for distance in distances:
    inp = f"action: continue distance: {distance}"
    targets = [
        f"Continue straight for {distance}.",
        f"Walk straight ahead for {distance}.",
        f"Keep going forward for {distance}.",
    ]
    for t in targets:
        dataset.append({"input": inp, "target": t})

# --- Arrival instructions ---
for landmark in landmarks:
    inp = f"action: arrive landmark: {landmark}"
    targets = [
        f"You have arrived at the {landmark}.",
        f"Your destination, the {landmark}, is here.",
        f"You are now at the {landmark}.",
    ]
    for t in targets:
        dataset.append({"input": inp, "target": t})

# --- Stop / warning instructions ---
for hazard in hazards:
    inp = f"action: stop reason: hazard description: {hazard}"
    targets = [
        f"Stop. There is a {hazard} ahead.",
        f"Please stop. Caution: {hazard}.",
        f"Stop immediately. {hazard.capitalize()} detected ahead.",
    ]
    for t in targets:
        dataset.append({"input": inp, "target": t})

# Shuffle to mix types
random.shuffle(dataset)

# Save as CSV
with open("nav_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input", "target"])
    writer.writeheader()
    writer.writerows(dataset)

print(f"Dataset created: {len(dataset)} examples saved to nav_dataset.csv")