import google.generativeai as genai
import time
import os
from pathlib import Path

def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue

        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load .env from project root first, then script directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_env_file(PROJECT_ROOT / ".env")
load_env_file(SCRIPT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Add it to your .env file.")

genai.configure(api_key=API_KEY)

# Use Gemini 1.5 Pro as it is best for following strict instructions and formatting
model = genai.GenerativeModel('gemini-3-flash-preview')

output_filename = "indoor_navigation_dataset.csv"
LAST_SCENARIO_INDEX = 47

# 2. Paste the finalized master prompt here
master_prompt = """
System Role: You are an expert dataset generator for machine learning models. 

Task: Generate a dataset of 50 unique navigation instructions for a seq2seq transformer model. This dataset trains an indoor navigation system to help visually impaired users safely navigate from a shopping mall's main entrance to various destinations using a static map.

Critical Dataset Rules & Minimalism:
* Strictly Action-Oriented: The user cannot see. Do NOT describe the environment. Do NOT mention passing by shops, glass doors, textures, or any landmarks along the way. 
* The "Only When Necessary" Rule: You may ONLY include a landmark in the input and target if it is one of two things:
  1. The final destination (e.g., "washroom", "Zara").
  2. A necessary floor-changing mechanism the user must interact with (e.g., "elevator", "escalator", "stairs").
* Unit of Measurement: Distances must be exclusively measured in steps (e.g., "15 steps"). 
* Format Constraints: Output the data strictly in a CSV format. Do not include any introductory or concluding text. Do not use markdown code blocks. Just output the raw CSV data.
* Column Structure: Two columns exactly: input and target.
* Input Column Format: action: [continue/turn/stop/board/exit] direction: [left/right/straight/up/down - if applicable] distance: [X steps - if applicable] landmark: [destination or floor-changer - ONLY if applicable]
* Target Column Format: The minimal, natural spoken English instruction wrapped in double quotes.

Scenario Focus for this Batch: Please focus all 50 generated records completely on this specific situation: [INSERT SCENARIO]

Examples to Learn From:
input,target
action: continue distance: 40 steps,"Continue straight for 40 steps."
action: turn direction: left distance: 15 steps,"Turn left and walk for 15 steps."
action: stop landmark: pharmacy,"Stop here. You have reached the pharmacy."
action: board direction: up landmark: escalator,"Step onto the escalator going up."
action: exit direction: straight distance: 10 steps landmark: elevator,"Exit the elevator and walk straight for 10 steps."
action: continue distance: 25 steps landmark: washroom,"Continue straight for 25 steps. The washroom is directly in front of you."
"""

# 3. Scenarios to cover
scenarios = [ 
    "Walking straight from the main entrance lobby deep into the ground floor.",
    "Navigating a simple L-shaped path: walking straight, making one 90-degree turn, and continuing.",
    "Making a series of alternating left and right turns through intersecting corridors without stopping.",
    "Walking a long distance and stopping exactly at a specific clothing store.",
    "Making a sharp U-turn (180 degrees) to walk down a parallel corridor.",
    "Stopping at a T-junction and turning right to continue walking.",
    "Navigating from the entrance directly to a ground-floor accessible restroom.",
    "Walking straight for a short distance and stopping at the mall management office.",
    "Making an immediate left turn right after passing the main entrance security check.",
    "A zig-zag path requiring precise step counts between multiple turns.",
    "Walking from the main entrance to locate the ground floor elevator bank.",
    "Entering an elevator and pressing the button for the 3rd floor.",
    "Exiting an elevator on the 2nd floor and walking straight ahead.",
    "Exiting an elevator and immediately turning left to begin walking.",
    "A full sequence: walking to the elevator, riding it down to the basement, and exiting.",
    "Navigating from the main entrance, taking an elevator up one floor, and stopping at a restaurant.",
    "Exiting a 4th-floor elevator and navigating two quick turns to reach the cinema.",
    "Finding a hidden service elevator at the end of a long, straight corridor.",
    "Exiting an elevator and arriving directly at the destination shop opposite the doors.",
    "Exiting an elevator and walking a long distance down a single straight corridor.",
    "Walking from the main entrance and stepping onto an ascending escalator.",
    "Stepping off an ascending escalator and walking straight for 20 steps.",
    "Stepping off a descending escalator and immediately turning right.",
    "Navigating from an entrance to an escalator, riding it up, and stopping at a pharmacy 5 steps away.",
    "Taking two consecutive escalators up (ground to 1st, then 1st to 2nd) with a short walk between them.",
    "Walking from a shop, turning left, and locating the descending escalator.",
    "Stepping off an escalator and navigating a zig-zag path to the restrooms.",
    "Locating a structural staircase and walking up one flight of stairs.",
    "Exiting a descending stairwell and making a sharp left turn.",
    "Walking down a long corridor to find the emergency exit stairs.",
    "Walking from the main entrance and making three consecutive right turns through connecting corridors.",
    "Navigating a rectangular loop: walking straight, turning right three times, and returning near the start point.",
    "Walking a long corridor, stopping at a T-junction, turning left, then immediately turning right into a side corridor.",
    "Making a sharp right turn immediately after entering the mall, then walking a short distance to a stop.",
    "Navigating a narrow service corridor requiring precise step counts between two sharp turns.",
    "Walking from the main entrance to the information kiosk located at the center of the ground floor.",
    "Walking straight from the entrance and stopping at the food court entrance on the ground floor.",
    "Navigating from the main entrance to a bank ATM located at the far end of a ground-floor corridor.",
    "Walking from the main entrance, turning once, and stopping at the ground-floor customer service desk.",
    "Navigating to a ground-floor pharmacy by making one turn midway through the corridor.",
    "Stepping off an ascending escalator, turning right, walking a short distance, and stopping at a bookstore.",
    "Riding a descending escalator to the basement level and walking straight to exit the escalator zone.",
    "Walking from a shop on the first floor, finding the escalator, and riding it up to the second floor.",
    "Stepping off an escalator on the third floor, making two turns, and stopping at a food court.",
    "Riding two consecutive descending escalators from the third floor to the ground floor with a short walk between them.",
    "Taking the elevator from the basement level up to the second floor and walking straight after exiting.",
    "Navigating from the main entrance to the elevator, riding to the fourth floor, making one turn, and stopping at a gym.",
    "Exiting an elevator on the third floor, turning right, walking a long corridor, and stopping at the cinema entrance.",
    "Taking the elevator from the first floor down to the basement and turning left immediately after exiting.",
    "Exiting an elevator and navigating a three-step path — straight, left turn, right turn — to reach a restroom.",
    "Walking up one flight of stairs and turning immediately right at the top landing.",
    "Walking down two flights of stairs from the second floor to the ground floor and stopping at the stairwell exit.",
    "Navigating from the main entrance to the staircase, climbing two floors, and stopping at the top landing.",
    "Exiting a stairwell on the first floor and walking a long straight corridor to reach a clothing store.",
    "Descending a staircase and making an immediate U-turn to walk along the ground-floor corridor.",
    "Taking an escalator up one floor and then navigating two turns to reach the restrooms.",
    "Riding an elevator to the second floor, exiting, walking straight, and stopping at a furniture store.",
    "Walking to the stairs, climbing one floor, turning left, and stopping at the nearest restroom.",
    "Navigating to the basement using the elevator and walking a straight corridor to the supermarket entrance.",
    "Taking an escalator up, making one turn, walking a defined number of steps, and stopping at a sports store.",
    "Stepping onto a descending elevator to go to the basement level.",
    "Boarding a downward escalator from the second floor to the first floor.",
    "Finding the descending staircase and beginning to walk down one floor.",
    "Stepping onto a descending escalator from the third floor to the second floor.",
    "Boarding the service elevator going down from the third floor to the ground floor.",
    "Walking to the emergency exit stairs and descending one flight down.",
    "Stepping onto a descending escalator immediately after exiting a shop.",
    "Boarding the elevator and pressing the button to go down to the basement parking.",
    "Walking from a restaurant to the nearest downward escalator and boarding it.",
    "Finding the staircase at the end of a corridor and walking down two flights.",
    "Locating the nearest staircase from the food court and beginning to ascend.",
    "Walking to the service elevator, boarding it, and going up to the third floor.",
    "Finding the emergency exit stairs at the corridor end and climbing one floor.",
    "Boarding the service elevator to descend from the fourth floor to the second.",
    "Walking a long corridor and boarding the structural staircase to go up one level.",
    "Locating the service elevator near the loading bay and riding it up one floor.",
    "Finding the emergency stairwell and descending from the second floor to ground.",
    "Boarding the stairs at the center of the mall and ascending two flights.",
    "Exiting an elevator and turning immediately left toward the food court.",
    "Stepping off an escalator and turning left to walk toward the pharmacy.",
    "Exiting the elevator on the second floor and turning left to find the restroom.",
    "Stepping off a descending escalator and turning left to locate the ATM.",
    "Exiting a stairwell and turning left to walk down the main corridor.",
    "Stepping off an ascending escalator and turning left to reach a clothing store.",
    "Exiting the service elevator and turning left to reach the loading area.",
    "Stepping off an escalator and turning left to stop at the information kiosk.",
    "Exiting the elevator on the basement level and turning left toward the supermarket.",
    "Stepping off emergency exit stairs and turning left into the main corridor.",
    "Exiting the elevator and turning right to walk toward the cinema entrance.",
    "Stepping off a descending escalator and turning right toward the customer service desk.",
    "Exiting the stairwell on the third floor and turning right to find the gym.",
    "Stepping off an ascending escalator and turning right toward the bookstore.",
    "Exiting the elevator on the fourth floor and turning right to reach the food court.",
    "Stepping off the emergency stairs and turning right into the shopping corridor.",
    "Exiting the service elevator and turning right toward the management office.",
    "Stepping off a descending escalator and turning right to stop at the pharmacy.",
    "Walking a very long straight corridor of over 80 steps from the main entrance deep into the mall.",
    "Continuing straight for more than 100 steps along the main ground floor corridor toward the far end.",
    "Walking a long uninterrupted straight path of around 70 steps to reach the elevator bank.",
    "Navigating a lengthy 90-step straight corridor on the second floor to reach a department store.",
    "Walking a 120-step corridor from one wing of the mall to the opposite wing without turning.",
    "Continuing straight along the basement corridor for about 80 steps to reach the supermarket.",
    "Walking a long 65-step path on the third floor from the escalator to the cinema.",
    "Navigating a straight 150-step path along the outer ring corridor of the mall.",
    "Continuing for 75 steps along the first-floor corridor to reach the food court entrance.",
    "Walking a 110-step straight path from the service elevator to the far end emergency exit.",
    "Stopping abruptly mid-corridor due to reaching a pre-defined waypoint with no named landmark.",
    "Stopping at the exact midpoint of a long corridor as an intermediate navigation checkpoint.",
    "Stopping immediately after completing a turn, before the next instruction is issued.",
    "Stopping at the end of a corridor where no named landmark exists but the path ends.",
    "Pausing at a corridor junction to await the next navigation instruction.",
    "Walking straight along a wide corridor and stopping at a named clothing store after explicitly confirming the straight direction.",
    "Continuing straight through a long wing of the mall and stopping at a pharmacy after an explicit direction confirmation.",
    "Walking explicitly straight for a long distance to reach the far elevator bank.",
    "Proceeding straight down a basement corridor explicitly to reach the supermarket entrance.",
    "Continuing straight on the fourth floor explicitly toward the cinema after exiting the escalator.",
    "Boarding an escalator going up, stepping off, immediately turning left, and stopping at a bookstore.",
    "Turning right from a junction, walking a long 80-step corridor, and stopping at the elevator bank.",
    "Exiting the elevator, turning right, walking a short distance, and stopping at the nearest restroom.",
    "Boarding a descending escalator, stepping off, turning left, and walking to the food court.",
    "Turning left at a T-junction, walking a medium distance, then boarding the elevator going up.",
    "Walking a very long straight corridor, making one right turn, and stopping at the gym entrance.",
    "Exiting the stairwell, turning right, walking 40 steps, and stopping at the cinema.",
    "Stepping off an escalator going up, walking straight for 60 steps, and stopping at a department store.",
]

def generate_dataset():
    # Start fresh only when beginning from the first scenario; otherwise resume appending.
    if LAST_SCENARIO_INDEX == 0:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("input,target\n")

    total_scenarios = len(scenarios)

    if LAST_SCENARIO_INDEX < 0 or LAST_SCENARIO_INDEX >= total_scenarios:
        raise ValueError(
            f"LAST_SCENARIO_INDEX must be between 0 and {total_scenarios - 1}, got {LAST_SCENARIO_INDEX}."
        )
    
    for index, scenario in enumerate(scenarios[LAST_SCENARIO_INDEX:], start=LAST_SCENARIO_INDEX):
        print(f"Processing scenario {index + 1} of {total_scenarios}...")
        
        # Inject the current scenario into the master prompt
        current_prompt = master_prompt.replace("[INSERT SCENARIO]", scenario)
        
        try:
            # Call the Gemini API
            response = model.generate_content(current_prompt)
            output_text = response.text.strip()
            
            # Clean up potential markdown formatting the AI might add by mistake
            output_text = output_text.replace("```csv", "").replace("```", "").strip()
            
            # Split the text into lines
            lines = output_text.split('\n')
            
            # Remove the header line if the AI included it
            if lines and "input,target" in lines[0].lower().replace(" ", ""):
                lines = lines[1:]
                
            # Append the clean lines to our CSV file
            with open(output_filename, 'a', encoding='utf-8') as f:
                for line in lines:
                    if line.strip(): # Make sure we don't write empty lines
                        f.write(line.strip() + "\n")
            
            print(f"Successfully added data for scenario {index + 1}.")
            
            # Pause for 30 seconds to avoid hitting API rate limits
            time.sleep(45)
            
        except Exception as e:
            print(f"An error occurred on scenario {index + 1}: {e}")

if __name__ == "__main__":
    print("Starting dataset generation...")
    generate_dataset()
    print(f"Finished! Your dataset is saved as {output_filename}")