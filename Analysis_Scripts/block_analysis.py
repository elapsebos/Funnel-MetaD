#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Perform Block Averaging Analysis.")
    parser.add_argument("-i", "--input", required=True, help="Input file or directory.")
    parser.add_argument("--filename", default="deltaG.txt", help="Default filename if directory is provided.")
    # Default reads 2nd column (index 1)
    parser.add_argument("-c", "--column", type=int, default=1, help="Column index to analyze (0-based).")
    parser.add_argument("-s", "--skip", type=int, default=0, help="Data points to skip (equilibration).")
    # Limit max block size to 50% of total length to avoid high variance at large blocks
    parser.add_argument("--max_block_frac", type=float, default=0.5, help="Max block size fraction.")
    parser.add_argument("-o", "--output", default="block_analysis.png", help="Output filename.")
    parser.add_argument("--title", default="Block Averaging Analysis", help="Plot title.")
    parser.add_argument("--ylim_min", type=float, help="Y-axis min.")
    parser.add_argument("--ylim_max", type=float, help="Y-axis max.")
    
    return parser.parse_args()

def get_block_error_overlapping(data, max_block_frac=0.5):
    """
    Calculates Block Averaging Error using a Sliding Window (Overlapping) algorithm.
    This eliminates 'jagged' plots caused by data discarding in non-overlapping methods.
    """
    N = len(data)
    if N < 50:
        print(f"Warning: Only {N} data points. Results unreliable.")
    
    # Set block size limits
    max_block_size = int(N * max_block_frac)
    min_block_size = 1
    
    # Sampling for plot smoothness (e.g., 100 points)
    steps = 100 
    block_sizes = np.unique(np.linspace(min_block_size, max_block_size, steps, dtype=int))
    block_sizes = block_sizes[block_sizes > 0] 
    
    errors = []
    valid_sizes = []

    data = np.array(data, dtype=float)

    print(f"Calculating overlapping blocks (Total frames: {N})...")
    
    for B in block_sizes:
        # 1. Moving Average via Convolution
        # mode='valid' means only fully overlapping windows are kept
        window = np.ones(B) / B
        moving_averages = np.convolve(data, window, mode='valid')
        
        # 2. Standard Deviation of the Means
        sigma_B = np.std(moving_averages, ddof=1)
        
        # 3. Standard Error of the Mean (SEM) Correction
        # Correction factor for dependent overlapping windows: SEM = sigma_B * sqrt(B / N)
        stat_error = sigma_B * np.sqrt(B / N)
        
        valid_sizes.append(B)
        errors.append(stat_error)
        
    return np.array(valid_sizes), np.array(errors)

def main():
    args = parse_args()
    
    # 1. Path Handling
    file_path = args.input
    if os.path.isdir(args.input):
        file_path = os.path.join(args.input, args.filename)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Reading {file_path}...")
    try:
        # Force reading only the specified column
        raw_data = np.loadtxt(file_path, comments=['#', '@', '!'], usecols=(args.column,))
    except Exception as e:
        print(f"Error: {e}")
        return

    data = raw_data

    # 2. Skip Equilibration
    if args.skip > 0:
        if args.skip >= len(data):
            print(f"Error: Skip ({args.skip}) >= Data Length ({len(data)}). No data left!")
            return
        data = data[args.skip:]
    
    print(f"Analyzing {len(data)} data points...")

    # 3. Calculate Errors
    sizes, errors = get_block_error_overlapping(data, args.max_block_frac)
    
    if len(errors) == 0:
        print("Error: Could not calculate errors (not enough data).")
        return

    # 4. Find Plateau
    plateau_error = np.max(errors)
    print(f"Estimated Statistical Error (Plateau): {plateau_error:.4f} kcal/mol")

    # 5. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, errors, marker='x', linestyle='-', color='tomato', label='Standard Error', markersize=5, linewidth=1)
    plt.axhline(y=plateau_error, color='red', linestyle='--', linewidth=1.5, label=f'Plateau: {plateau_error:.3f}')
    
    plt.xlabel('Block Size', fontsize=12)
    plt.ylabel(r'Error (kcal mol$^{-1}$)', fontsize=12)
    plt.title(args.title, fontsize=14)
    
    if args.ylim_min is not None: plt.ylim(bottom=args.ylim_min)
    if args.ylim_max is not None: plt.ylim(top=args.ylim_max)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    print(f"Saving to {args.output}")
    plt.savefig(args.output, dpi=300)

if __name__ == "__main__":
    main()