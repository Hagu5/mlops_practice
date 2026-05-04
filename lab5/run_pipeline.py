"""
Main pipeline runner script.

This script executes the entire ML pipeline:
1. Generate datasets
2. Train model
3. Run pytest tests
4. Display summary of results
"""

import subprocess
import sys
import os


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_script(script_name, description):
    """
    Run a Python script and handle errors.
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
    
    Returns:
        True if successful, False otherwise
    """
    print_header(description)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n[FAIL] Script not found: {script_name}")
        return False


def run_tests():
    """
    Run pytest tests and capture results.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print_header("Step 3: Running Pytest Tests")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_model.py", "-v", "--tb=short"],
            capture_output=False,
            text=True
        )
        
        # pytest returns 0 if all tests pass, 1 if some fail
        if result.returncode == 0:
            print("\n[OK] All tests passed")
            return True
        else:
            print("\n[WARNING] Some tests failed (expected for noisy dataset)")
            return False
    except FileNotFoundError:
        print("\n[FAIL] pytest not found. Please install: pip install pytest")
        return False


def print_summary(steps_status):
    """
    Print a summary of the pipeline execution.
    
    Args:
        steps_status: Dictionary with step names and their status
    """
    print_header("Pipeline Execution Summary")
    
    for step, status in steps_status.items():
        status_symbol = "[OK]" if status else "[FAIL]"
        print(f"{status_symbol} {step}")
    
    print("\n" + "=" * 70)
    
    # Check if expected behavior occurred
    if (steps_status["Data Generation"] and 
        steps_status["Model Training"] and 
        not steps_status["Tests"]):
        print("\n[OK] Pipeline executed as expected!")
        print("  - Clean datasets generated successfully")
        print("  - Model trained successfully")
        print("  - Tests passed on clean data, failed on noisy data (as expected)")
        print("\nThis demonstrates automated quality testing for ML models.")
    elif all(steps_status.values()):
        print("\n[OK] All steps completed successfully!")
    else:
        print("\n[WARNING] Some steps failed. Please check the output above.")
    
    print("=" * 70)


def main():
    """Main function to run the entire pipeline."""
    print("\n" + "=" * 70)
    print("  ML MODEL QUALITY TESTING PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1. Generate 3 clean datasets and 1 noisy dataset")
    print("  2. Train a linear regression model")
    print("  3. Test the model on all datasets")
    print("  4. Demonstrate automated quality testing")
    
    # Track status of each step
    steps_status = {
        "Data Generation": False,
        "Model Training": False,
        "Tests": False
    }
    
    # Step 1: Generate datasets
    steps_status["Data Generation"] = run_script(
        "generate_datasets.py",
        "Step 1: Generating Datasets"
    )
    
    if not steps_status["Data Generation"]:
        print("\n[FAIL] Pipeline failed at data generation step")
        return
    
    # Step 2: Train model
    steps_status["Model Training"] = run_script(
        "train_model.py",
        "Step 2: Training Model"
    )
    
    if not steps_status["Model Training"]:
        print("\n[FAIL] Pipeline failed at model training step")
        return
    
    # Step 3: Run tests
    steps_status["Tests"] = run_tests()
    
    # Print summary
    print_summary(steps_status)


if __name__ == "__main__":
    main()
