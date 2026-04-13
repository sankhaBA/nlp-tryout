import google.generativeai as genai
import time
import os
import json
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
output_path = SCRIPT_DIR / output_filename
metadata_filename = f"{output_path.stem}_metadata.json"
metadata_path = SCRIPT_DIR / metadata_filename
master_prompt_path = SCRIPT_DIR / "master_prompt.txt"
scenarios_path = SCRIPT_DIR / "scenarios.json"
LAST_SCENARIO_INDEX = 63

def load_master_prompt(prompt_path):
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    return prompt_path.read_text(encoding='utf-8').strip()


def load_scenarios(scenarios_file_path):
    if not scenarios_file_path.exists():
        raise FileNotFoundError(f"Missing scenarios file: {scenarios_file_path}")

    payload = json.loads(scenarios_file_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Scenarios file must contain a non-empty JSON array: {scenarios_file_path}")
    if not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError(f"Every scenario must be a non-empty string: {scenarios_file_path}")

    return payload


master_prompt = load_master_prompt(master_prompt_path)
scenarios = load_scenarios(scenarios_path)

def generate_dataset():
    def get_start_scenario_index_from_metadata():
        if not metadata_path.exists():
            return None

        try:
            payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Warning: could not parse metadata file at {metadata_path}: {e}")
            return None

        last_success = payload.get("last_successful_scenario_index")
        if isinstance(last_success, int) and last_success >= 1:
            # Metadata stores 1-based index of last successful scenario.
            # The next zero-based start index is numerically the same value.
            return last_success

        return None

    def write_metadata(last_success_index, last_success_scenario, status, failed_index=None, failed_scenario=None, error_message=None):
        payload = {
            "dataset_file": output_filename,
            "last_successful_scenario_index": last_success_index,
            "last_successful_scenario_text": last_success_scenario,
            "status": status,
        }

        if failed_index is not None:
            payload["failed_scenario_index"] = failed_index
        if failed_scenario is not None:
            payload["failed_scenario_text"] = failed_scenario
        if error_message is not None:
            payload["error_message"] = error_message

        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

    metadata_start_index = get_start_scenario_index_from_metadata()
    start_scenario_index = metadata_start_index if metadata_start_index is not None else LAST_SCENARIO_INDEX

    if metadata_start_index is not None:
        print(f"Resuming from metadata at scenario index {start_scenario_index}.")
    else:
        print(f"Using configured LAST_SCENARIO_INDEX={LAST_SCENARIO_INDEX}.")

    # Start fresh only when beginning from the first scenario; otherwise resume appending.
    if start_scenario_index == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("input,target\n")

    total_scenarios = len(scenarios)
    last_successful_scenario_index = None
    last_successful_scenario_text = None
    stopped_on_error = False

    if start_scenario_index < 0 or start_scenario_index > total_scenarios:
        raise ValueError(
            f"Resolved start scenario index must be between 0 and {total_scenarios}, got {start_scenario_index}."
        )
    
    for index, scenario in enumerate(scenarios[start_scenario_index:], start=start_scenario_index):
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
            with open(output_path, 'a', encoding='utf-8') as f:
                for line in lines:
                    if line.strip(): # Make sure we don't write empty lines
                        f.write(line.strip() + "\n")

            last_successful_scenario_index = index + 1
            last_successful_scenario_text = scenario
            write_metadata(
                last_successful_scenario_index,
                last_successful_scenario_text,
                status="running"
            )
            
            print(f"Successfully added data for scenario {index + 1}.")
            
            # Pause for 30 seconds to avoid hitting API rate limits
            time.sleep(45)
            
        except Exception as e:
            print(f"An error occurred on scenario {index + 1}: {e}")
            write_metadata(
                last_successful_scenario_index,
                last_successful_scenario_text,
                status="stopped_on_error",
                failed_index=index + 1,
                failed_scenario=scenario,
                error_message=str(e)
            )
            stopped_on_error = True
            break

    if not stopped_on_error:
        write_metadata(
            last_successful_scenario_index,
            last_successful_scenario_text,
            status="completed"
        )

if __name__ == "__main__":
    print("Starting dataset generation...")
    generate_dataset()
    print(f"Finished! Your dataset is saved as {output_path}")
    print(f"Run metadata is saved as {metadata_path}")