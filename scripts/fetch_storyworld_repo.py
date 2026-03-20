from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url', required=True)
    parser.add_argument('--dest', default='external/GPTStoryworld')
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        print(f'Repo destination already exists and is non-empty: {dest.resolve()}')
        return

    subprocess.run(['git', 'clone', args.repo_url, str(dest)], check=True)
    print(f'Cloned to {dest.resolve()}')


if __name__ == '__main__':
    main()
