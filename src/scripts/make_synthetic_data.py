import subprocess
import sys
import os

os.chdir(os.path.dirname(__file__))

def main():
    alpha, beta, min_samples, max_samples, name, num_users = sys.argv[1:7]
    print("generate_synthetic.py")
    subprocess.run([
        "python", "../../FedProx/generate_synthetic.py"] + sys.argv[1:])
    print("extract_synthetic_data.py")
    subprocess.run(["python", "extract_synthetic_data.py", f"{alpha}_{beta}_{name}"])
    print("get_data_stats.py")
    subprocess.run(["python", "get_data_stats.py", f"synthetic_{alpha}_{beta}_{name}"])

if __name__ == "__main__":
    main()