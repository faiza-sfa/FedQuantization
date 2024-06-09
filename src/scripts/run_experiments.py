import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

def main():
    experiments = sys.argv[1:]
    for experiment in experiments:
        print(f"Running experiment {experiment}")
        subprocess.run(["python", "run_experiment.py", experiment])

if __name__ == "__main__":
    main()