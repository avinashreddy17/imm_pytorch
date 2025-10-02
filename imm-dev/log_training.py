#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Logging Script
Captures all terminal output during training with timestamps
"""

import os
import sys
import subprocess
import datetime
import time

def setup_logging():
    """Setup logging directory and filename"""
    log_dir = "data/logs/debug_perceptual"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, "training_log_{}.txt".format(timestamp))
    
    return log_file

def run_training_with_logging(log_file):
    """Run training and capture all output"""
    print("=" * 80)
    print("STARTING TRAINING WITH LOGGING")
    print("=" * 80)
    print("Log file: {}".format(log_file))
    print("Start time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80)
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("IMM DEBUG TRAINING LOG\n")
        f.write("=" * 80 + "\n")
        f.write("Start time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Command: python debug_train_perceptual.py\n")
        f.write("=" * 80 + "\n\n")
        f.flush()
        
        # Run training command
        try:
            process = subprocess.Popen(
                [sys.executable, "debug_train_perceptual.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both terminal and file
            for line in iter(process.stdout.readline, ''):
                # Print to terminal
                print(line.rstrip())
                # Write to file
                f.write(line)
                f.flush()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRAINING COMPLETED\n")
            f.write("End time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("Return code: {}\n".format(return_code))
            f.write("=" * 80 + "\n")
            
            return return_code
            
        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("TRAINING INTERRUPTED BY USER")
            print("=" * 80)
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRAINING INTERRUPTED BY USER\n")
            f.write("End time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("=" * 80 + "\n")
            return 1
        except Exception as e:
            print("\n" + "=" * 80)
            print("ERROR DURING TRAINING: {}".format(str(e)))
            print("=" * 80)
            f.write("\n" + "=" * 80 + "\n")
            f.write("ERROR DURING TRAINING: {}\n".format(str(e)))
            f.write("End time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("=" * 80 + "\n")
            return 1

def main():
    """Main function"""
    print("IMM Debug Training with Logging")
    print("This will capture all terminal output to a log file")
    print()
    
    # Setup logging
    log_file = setup_logging()
    
    # Confirm before starting
    response = raw_input("Start training with logging? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    return_code = run_training_with_logging(log_file)
    
    print("\n" + "=" * 80)
    print("TRAINING SESSION COMPLETED")
    print("Log saved to: {}".format(log_file))
    print("=" * 80)
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())


