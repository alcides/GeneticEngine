import sys
import subprocess

if __name__ == "__main__":
    if '-install-packages' in sys.argv:
        print("Installing packages")
        print("====================================\n")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'src\.'])

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("------------------------------------")
        print("Packages installed")
        print("------------------------------------")

    succesful_examples = []
    failed_examples = []

    def run_example(file_name):
        print(f"\n\nRunning {file_name} example:")
        print("------------------------------------\n")

        try:
            output = subprocess.check_output(['python', f'examples\\{file_name}'], stderr=subprocess.STDOUT)
            succesful_examples.append(f'{file_name}')
        except subprocess.CalledProcessError as exc:
            print(exc.output)
            failed_examples.append((f'{file_name}',exc))
        

    print("Running examples:")
    print("====================================\n")
    
    run_example("example.py")
    
    run_example("pymax.py")

    run_example("vectorialgp_example.py")

    run_example("regression_example.py")

    run_example("santafe.py")

    print("\nSuccesfull examples:\n--------")
    for example in succesful_examples:
        print(f"\t - {example}")
    print("\nFailed examples:\n--------")
    for example, error in failed_examples:
        print(f"\t - {example} gave error: {error.output}")

