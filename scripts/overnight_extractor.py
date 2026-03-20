import argparse
import json
import os
import sys
import time
import uuid
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.logger import JsonlLogger
from harness.model_client import OpenAICompatibleClient
from harness.runner import EpisodeRunner
from envs.registry import build_env

# Large list of reasoning tasks
REASONING_TASKS = [
    'gsm8k', 'logic_puzzle', 'base_conversion', 'caesar_cipher', 'cryptarithm',
    'game_of_life', 'graph_color', 'jugs', 'letter_counting', 'word_ladder',
    'sudoku', 'tower_of_hanoi', 'n_queens', 'circuit_logic', 'knights_knaves',
    'syllogism', 'zebra_puzzles', 'coin_flip', 'simple_equations', 'simple_integration'
]

def get_env_spec(category, task_name, seed):
    if category == 'reasoning_gym':
        return {
            'type': 'reasoning_gym',
            'name': f"rg_{task_name}",
            'gym_config': task_name,
            'num_examples': 1,
            'seed': seed
        }
    if category == 'storyworld':
        return {
            'type': 'storyworld_native',
            'name': "storyworld_overnight",
            'repo_path': "C:/projects/GPTStoryworld",
            'storyworld_path': "C:/projects/GPTStoryworld/raw_storyworld.json",
            'seed': seed,
            'log_path': f"data/overnight/storyworld_env_logs.jsonl"
        }
    if category == 'sweepweave':
        return {
            'type': 'sweepweave_prime',
            'name': "sweepweave_overnight",
            'repo_path': "C:/projects/prime-environments",
            'load_kwargs': {
                'num_examples': 1,
                'seed': seed,
                'max_turns': 10
            }
        }
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://snacksack-ms-7d32.tail3156cd.ts.net:8080/v1")
    parser.add_argument("--model", default="Qwen3.5-27B.Q4_K_M.gguf")
    parser.add_argument("--duration-hours", type=float, default=8.0)
    args = parser.parse_args()

    model = OpenAICompatibleClient(
        base_url=args.url,
        model_name=args.model,
        max_new_tokens=1024,
        temperature=0.8, # Increase for more diverse/novel data
        top_p=0.95
    )

    start_time = time.time()
    end_time = start_time + (args.duration_hours * 3600)
    
    # Use consolidated files instead of thousands of tiny ones
    out_dir = Path("data/overnight")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extraction starting. End time: {datetime.fromtimestamp(end_time)}")

    categories = ['reasoning_gym', 'storyworld', 'sweepweave']
    
    while time.time() < end_time:
        category = random.choice(categories)
        seed = random.randint(1, 1000000)
        
        task_name = random.choice(REASONING_TASKS) if category == 'reasoning_gym' else category
        spec = get_env_spec(category, task_name, seed)
        
        log_file = out_dir / f"{category}_consolidated.jsonl"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Category={category} Task={task_name} Seed={seed}")
        
        try:
            logger = JsonlLogger(str(log_file))
            runner = EpisodeRunner(model, logger)
            env = build_env(spec)
            
            max_steps = 15 if category != 'reasoning_gym' else 1
            records = runner.run_episode(env, max_steps=max_steps)
            
            if records:
                # Check if we actually got content
                sample = records[0]
                content_len = len(sample.thought or "") + len(sample.action or "")
                print(f"  Captured {len(records)} steps. Content length: {content_len} chars.")
            
            # Substantial delay to let the 27B model breathe and ensure port doesn't lock up
            time.sleep(5)
            
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)

    print("Done.")

if __name__ == "__main__":
    main()
