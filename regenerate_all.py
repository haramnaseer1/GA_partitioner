import os
import subprocess
import sys

def run_command(command):
    """Runs a command in the shell and prints its output."""
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        if rc != 0:
            print(f"Error executing command: {command}. Return code: {rc}")
        return rc
    except Exception as e:
        print(f"An exception occurred: {e}")
        return -1

def main():
    """
    Main function to find all application json files and run the GA pipeline for each.
    """
    application_dir = "Application"
    solution_dir = "solution"
    
    if not os.path.isdir(application_dir):
        print(f"Error: Application directory '{application_dir}' not found.")
        sys.exit(1)

    if not os.path.exists(solution_dir):
        os.makedirs(solution_dir)
        print(f"Created solution directory: {solution_dir}")

    application_files = [f for f in os.listdir(application_dir) if f.endswith('.json')]
    
    if not application_files:
        print(f"No JSON files found in '{application_dir}'.")
        return

    print(f"Found {len(application_files)} application files to process.")

    for i, filename in enumerate(application_files):
        app_file_path = os.path.join(application_dir, filename)
        solution_file_name = f"{os.path.splitext(filename)[0]}_ga.json"
        solution_file_path = os.path.join(solution_dir, solution_file_name)

        print(f"--- Processing file {i+1}/{len(application_files)}: {filename} ---")

        # 1. Run the main GA partitioner
        main_command = f"python -m src.main 0 {app_file_path}"
        if run_command(main_command) != 0:
            print(f"Skipping remaining steps for {filename} due to error in main GA script.")
            continue

        # 2. Simplify the output
        simplify_command = f"python src/simplify.py --input {app_file_path}"
        if run_command(simplify_command) != 0:
            print(f"Skipping validation for {filename} due to error in simplify script.")
            continue
            
        # 3. Check the solution
        check_command = f"python Script/check_solutions.py --solution {solution_file_path}"
        run_command(check_command)

        print(f"--- Finished processing {filename} ---")

    print("All application files have been processed.")

if __name__ == "__main__":
    main()
