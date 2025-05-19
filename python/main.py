# run_all.py
import subprocess
import sys
import os

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n🚀 Running {script_name} ...\n")

    try:
        subprocess.run(
            [sys.executable, "-u", script_path],  # "-u" ensures unbuffered output
            check=True
        )
        print(f"\n✅ Finished {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name} (exit code {e.returncode})")
        sys.exit(e.returncode)

if __name__ == "__main__":
    run_script("generate_dataset.py")
    run_script("train_model.py")
