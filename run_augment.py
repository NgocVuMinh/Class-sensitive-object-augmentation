import subprocess

def run_n_times(suffix):
    # Call script and pass the number of time to run
    subprocess.run(["python", "augmentation.py", "--suffix", str(suffix)])

def main():
    # The number of times to run augmentation
    n = 10  
    for i in range(1, n+1):
        run_n_times(i)

if __name__ == "__main__":
    main()
