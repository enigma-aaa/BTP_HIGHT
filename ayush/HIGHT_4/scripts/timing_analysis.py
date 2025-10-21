#!/usr/bin/env python3
"""
Timing Analysis Script for HIGHT Pretraining
Analyzes timing data from training logs and provides detailed breakdown
"""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from pathlib import Path

class PretrainingTimingAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.timing_data = {}
        self.step_times = []
        
    def parse_log_file(self):
        """Parse the training log file to extract timing information"""
        print(f"Parsing log file: {self.log_file_path}")
        
        with open(self.log_file_path, 'r') as f:
            content = f.read()
        
        # Extract different timing phases
        self._extract_model_loading_times(content)
        self._extract_training_step_times(content)
        self._extract_memory_usage_times(content)
        self._extract_optimizer_times(content)
        
    def _extract_model_loading_times(self, content):
        """Extract model loading timing information"""
        # Model loading checkpoint shards timing
        checkpoint_pattern = r'Loading checkpoint shards:\s+(\d+)%\|\s*([█\s]+)\|\s*(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)s/it\]'
        matches = re.findall(checkpoint_pattern, content)
        
        if matches:
            self.timing_data['model_loading'] = {
                'checkpoint_shards': []
            }
            for match in matches:
                progress, bar, current, total, elapsed_min, elapsed_sec, remaining_min, remaining_sec, time_per_item = match
                self.timing_data['model_loading']['checkpoint_shards'].append({
                    'progress': int(progress),
                    'current': int(current),
                    'total': int(total),
                    'elapsed_seconds': int(elapsed_min) * 60 + int(elapsed_sec),
                    'remaining_seconds': int(remaining_min) * 60 + int(remaining_sec),
                    'time_per_item': float(time_per_item)
                })
    
    def _extract_training_step_times(self, content):
        """Extract training step timing information"""
        # Training step progress timing
        step_pattern = r'\s+(\d+)%\|\s*([█\s]+)\|\s*(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)s/it\]'
        matches = re.findall(step_pattern, content)
        
        if matches:
            self.timing_data['training_steps'] = []
            for match in matches:
                progress, bar, current, total, elapsed_min, elapsed_sec, remaining_min, remaining_sec, time_per_step = match
                self.step_times.append(float(time_per_step))
                self.timing_data['training_steps'].append({
                    'step': int(current),
                    'total_steps': int(total),
                    'progress': int(progress),
                    'elapsed_seconds': int(elapsed_min) * 60 + int(elapsed_sec),
                    'remaining_seconds': int(remaining_min) * 60 + int(remaining_sec),
                    'time_per_step': float(time_per_step)
                })
    
    def _extract_memory_usage_times(self, content):
        """Extract memory usage timing information"""
        # Memory usage patterns
        memory_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\s+\[INFO\]\s+\[utils\.py:\d+:see_memory_usage\]\s+(Before|After)\s+(.*?)\s+MA\s+([\d.]+)\s+GB\s+Max_MA\s+([\d.]+)\s+GB\s+CA\s+([\d.]+)\s+GB\s+Max_CA\s+([\d.]+)\s+GB'
        matches = re.findall(memory_pattern, content)
        
        if matches:
            self.timing_data['memory_usage'] = []
            for match in matches:
                timestamp, phase, operation, ma, max_ma, ca, max_ca = match
                self.timing_data['memory_usage'].append({
                    'timestamp': timestamp,
                    'phase': phase,
                    'operation': operation,
                    'memory_allocated_gb': float(ma),
                    'max_memory_allocated_gb': float(max_ma),
                    'cache_allocated_gb': float(ca),
                    'max_cache_allocated_gb': float(max_ca)
                })
    
    def _extract_optimizer_times(self, content):
        """Extract optimizer initialization timing"""
        # DeepSpeed optimizer timing
        optimizer_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\s+\[INFO\].*?(Before|After)\s+(initializing optimizer states|initializing ZeRO optimizer)'
        matches = re.findall(optimizer_pattern, content)
        
        if matches:
            self.timing_data['optimizer_timing'] = []
            for match in matches:
                timestamp, phase, operation = match
                self.timing_data['optimizer_timing'].append({
                    'timestamp': timestamp,
                    'phase': phase,
                    'operation': operation
                })
    
    def generate_timing_report(self):
        """Generate a comprehensive timing report"""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'log_file': str(self.log_file_path),
            'summary': {},
            'detailed_analysis': {}
        }
        
        # Model loading analysis
        if 'model_loading' in self.timing_data:
            checkpoint_data = self.timing_data['model_loading']['checkpoint_shards']
            if checkpoint_data:
                total_loading_time = checkpoint_data[-1]['elapsed_seconds']
                report['summary']['model_loading_time_seconds'] = total_loading_time
                report['summary']['model_loading_time_minutes'] = total_loading_time / 60
                report['detailed_analysis']['model_loading'] = checkpoint_data
        
        # Training step analysis
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            min_step_time = min(self.step_times)
            max_step_time = max(self.step_times)
            
            report['summary']['average_step_time_seconds'] = avg_step_time
            report['summary']['min_step_time_seconds'] = min_step_time
            report['summary']['max_step_time_seconds'] = max_step_time
            report['summary']['total_training_steps'] = len(self.step_times)
            
            if 'training_steps' in self.timing_data:
                last_step = self.timing_data['training_steps'][-1]
                report['summary']['total_elapsed_time_seconds'] = last_step['elapsed_seconds']
                report['summary']['estimated_remaining_time_seconds'] = last_step['remaining_seconds']
        
        # Memory usage analysis
        if 'memory_usage' in self.timing_data:
            memory_data = self.timing_data['memory_usage']
            peak_memory = max([m['max_memory_allocated_gb'] for m in memory_data])
            report['summary']['peak_memory_usage_gb'] = peak_memory
            report['detailed_analysis']['memory_usage'] = memory_data
        
        return report
    
    def create_visualizations(self, output_dir="timing_analysis_output"):
        """Create timing visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step time progression plot
        if self.step_times:
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(self.step_times) + 1), self.step_times, 'b-', linewidth=2)
            plt.title('Training Step Time Progression')
            plt.xlabel('Step Number')
            plt.ylabel('Time per Step (seconds)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'step_time_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Step time histogram
            plt.figure(figsize=(10, 6))
            plt.hist(self.step_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of Step Times')
            plt.xlabel('Time per Step (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'step_time_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Memory usage plot
        if 'memory_usage' in self.timing_data:
            memory_data = self.timing_data['memory_usage']
            timestamps = [datetime.strptime(m['timestamp'], '%Y-%m-%d %H:%M:%S') for m in memory_data]
            memory_values = [m['memory_allocated_gb'] for m in memory_data]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, memory_values, 'g-', linewidth=2, marker='o', markersize=4)
            plt.title('Memory Usage Over Time')
            plt.xlabel('Time')
            plt.ylabel('Memory Allocated (GB)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'memory_usage_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def save_report(self, report, output_file="timing_analysis_report.json"):
        """Save timing report to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Timing report saved to: {output_file}")
    
    def print_summary(self, report):
        """Print a human-readable summary"""
        print("\n" + "="*60)
        print("HIGHT PRETRAINING TIMING ANALYSIS SUMMARY")
        print("="*60)
        
        summary = report['summary']
        
        if 'model_loading_time_minutes' in summary:
            print(f"Model Loading Time: {summary['model_loading_time_minutes']:.2f} minutes")
        
        if 'average_step_time_seconds' in summary:
            print(f"Average Step Time: {summary['average_step_time_seconds']:.2f} seconds")
            print(f"Min Step Time: {summary['min_step_time_seconds']:.2f} seconds")
            print(f"Max Step Time: {summary['max_step_time_seconds']:.2f} seconds")
        
        if 'total_training_steps' in summary:
            print(f"Total Training Steps Completed: {summary['total_training_steps']}")
        
        if 'total_elapsed_time_seconds' in summary:
            elapsed_minutes = summary['total_elapsed_time_seconds'] / 60
            print(f"Total Elapsed Time: {elapsed_minutes:.2f} minutes")
        
        if 'estimated_remaining_time_seconds' in summary:
            remaining_minutes = summary['estimated_remaining_time_seconds'] / 60
            print(f"Estimated Remaining Time: {remaining_minutes:.2f} minutes")
        
        if 'peak_memory_usage_gb' in summary:
            print(f"Peak Memory Usage: {summary['peak_memory_usage_gb']:.2f} GB")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze HIGHT pretraining timing')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('--output-dir', default='timing_analysis_output', 
                       help='Output directory for analysis results')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PretrainingTimingAnalyzer(args.log_file)
    
    # Parse log file
    analyzer.parse_log_file()
    
    # Generate report
    report = analyzer.generate_timing_report()
    
    # Print summary
    analyzer.print_summary(report)
    
    # Save report
    analyzer.save_report(report, f"{args.output_dir}/timing_report.json")
    
    # Create visualizations
    if not args.no_plots:
        analyzer.create_visualizations(args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
