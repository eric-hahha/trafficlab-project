#!/usr/bin/env python3
"""
Trajectory Plotter for TrafficLab 3D (Command Line Version)

This script plots object trajectories onto a satellite image without GUI dependencies.
Usage: python trajectory_plotter_cli.py [path_to_json_gz_file]
"""

import os
import sys
import gzip
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class TrajectoryPlotterCLI:
    """Command-line trajectory plotter without GUI dependencies."""
    
    def __init__(self):
        self.data = None
        self.location_name = None
        self.sat_image = None
        self.sat_image_path = None
    
    def find_json_files(self) -> List[str]:
        """Find all .json.gz files in output directory."""
        json_files = []
        if os.path.exists("output"):
            for root, dirs, files in os.walk("output"):
                for file in files:
                    if file.endswith(('.json.gz', '.json')):
                        json_files.append(os.path.join(root, file))
        return sorted(json_files)
    
    def select_json_file(self, file_path: str = None) -> Optional[str]:
        """Select JSON file from command line argument or interactive selection."""
        if file_path and os.path.exists(file_path):
            return file_path
        
        json_files = self.find_json_files()
        
        if not json_files:
            print("No .json.gz files found in output folder.")
            return None
        
        if len(json_files) == 1:
            print(f"Found single trajectory file: {json_files[0]}")
            return json_files[0]
        
        print(f"\nFound {len(json_files)} trajectory files:")
        for i, file_path in enumerate(json_files, 1):
            rel_path = os.path.relpath(file_path)
            print(f"{i:2d}. {rel_path}")
        
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(json_files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(json_files):
                    return json_files[idx]
                else:
                    print(f"Please enter a number between 1 and {len(json_files)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def load_trajectory_data(self, file_path: str) -> bool:
        """Load trajectory data from .json.gz file."""
        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    raw_data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            
            # Extract frames from the data structure
            if isinstance(raw_data, dict) and 'frames' in raw_data:
                self.data = raw_data['frames']
                print(f"Loaded {len(self.data)} frames from 'frames' key")
            elif isinstance(raw_data, list):
                self.data = raw_data
                print(f"Loaded {len(self.data)} frames as direct list")
            else:
                print(f"Unexpected data structure: {type(raw_data)}")
                return False
            
            # Extract location name from file path
            path_parts = Path(file_path).parts
            for i, part in enumerate(path_parts):
                if part in ["test1", "119NH"] or part.endswith("NH"):
                    self.location_name = part
                    break
            
            if not self.location_name:
                filename = Path(file_path).stem.replace(".json", "")
                if "test1" in filename:
                    self.location_name = "test1"
                elif "119" in filename:
                    self.location_name = "119NH"
            
            print(f"Detected location: {self.location_name}")
            return True
            
        except Exception as e:
            print(f"Error: Failed to load trajectory data: {str(e)}")
            return False
    
    def load_satellite_image(self) -> bool:
        """Load the corresponding satellite image for the location."""
        if not self.location_name:
            print("Error: Could not determine location from file path")
            return False
        
        sat_image_path = f"location/{self.location_name}/sat_{self.location_name}.png"
        
        if not os.path.exists(sat_image_path):
            print(f"Error: Satellite image not found: {sat_image_path}")
            return False
        
        try:
            self.sat_image = Image.open(sat_image_path)
            self.sat_image_path = sat_image_path
            print(f"Loaded satellite image: {sat_image_path}")
            print(f"Image size: {self.sat_image.size}")
            return True
            
        except Exception as e:
            print(f"Error: Failed to load satellite image: {str(e)}")
            return False
    
    def extract_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Extract trajectories from the loaded data."""
        trajectories = {}
        
        for frame_data in self.data:
            if "objects" not in frame_data:
                continue
                
            for obj in frame_data["objects"]:
                tracked_id = obj.get("tracked_id")
                sat_coords = obj.get("sat_coords") or obj.get("sat_coord")
                
                if tracked_id is not None and sat_coords:
                    if tracked_id not in trajectories:
                        trajectories[tracked_id] = []
                    trajectories[tracked_id].append((sat_coords[0], sat_coords[1]))
        
        # Filter out trajectories with too few points
        min_points = 3
        filtered_trajectories = {
            tid: points for tid, points in trajectories.items() 
            if len(points) >= min_points
        }
        
        print(f"Extracted {len(filtered_trajectories)} trajectories")
        return filtered_trajectories
    
    def get_object_classes(self) -> Dict[int, str]:
        """Get object classes for each tracked ID."""
        object_classes = {}
        
        for frame_data in self.data:
            if "objects" not in frame_data:
                continue
                
            for obj in frame_data["objects"]:
                tracked_id = obj.get("tracked_id")
                obj_class = obj.get("class", "unknown")
                
                if tracked_id is not None:
                    object_classes[tracked_id] = obj_class
        
        return object_classes
    
    def get_color_for_id(self, tracked_id: int) -> str:
        """Get a unique color for each tracked ID using a color palette."""
        # Extended color palette for better distinction
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#85C1E9", "#F8C471",
            "#82E0AA", "#F1948A", "#BB8FCE", "#85C1E9", "#F9E79F",
            "#AED6F1", "#A9DFBF", "#F5B7B1", "#D2B4DE", "#AED6F1",
            "#FADBD8", "#D5F4E6", "#FCF3CF", "#EBDEF0", "#D6EAF8",
            "#E8F8F5", "#FEF9E7", "#F4ECF7", "#EBF5FB", "#E8F6F3"
        ]
        return colors[tracked_id % len(colors)]
    
    def select_trajectory_ids(self, trajectories: Dict[int, List[Tuple[float, float]]], 
                            object_classes: Dict[int, str]) -> List[int]:
        """Allow user to select which trajectory IDs to plot."""
        if not trajectories:
            return []
        
        print(f"\nFound {len(trajectories)} trajectories:")
        print("ID  | Class      | Points | Color Preview")
        print("-" * 45)
        
        sorted_ids = sorted(trajectories.keys())
        for tid in sorted_ids:
            obj_class = object_classes.get(tid, "unknown")
            point_count = len(trajectories[tid])
            color = self.get_color_for_id(tid)
            print(f"{tid:3d} | {obj_class:10s} | {point_count:6d} | {color}")
        
        print("\nSelection options:")
        print("1. Plot all trajectories")
        print("2. Select specific IDs")
        print("3. Select by class type")
        print("4. Select ID range")
        
        while True:
            try:
                choice = input("\nChoose option (1-4): ").strip()
                
                if choice == "1":
                    return sorted_ids
                
                elif choice == "2":
                    print(f"\nEnter trajectory IDs to plot (comma-separated):")
                    print(f"Available IDs: {', '.join(map(str, sorted_ids))}")
                    id_input = input("IDs: ").strip()
                    
                    if not id_input:
                        continue
                    
                    try:
                        selected_ids = [int(x.strip()) for x in id_input.split(',')]
                        valid_ids = [tid for tid in selected_ids if tid in trajectories]
                        
                        if valid_ids:
                            print(f"Selected {len(valid_ids)} trajectories: {valid_ids}")
                            return valid_ids
                        else:
                            print("No valid IDs found. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter numbers separated by commas.")
                
                elif choice == "3":
                    # Group by class
                    class_groups = {}
                    for tid in sorted_ids:
                        obj_class = object_classes.get(tid, "unknown")
                        if obj_class not in class_groups:
                            class_groups[obj_class] = []
                        class_groups[obj_class].append(tid)
                    
                    print(f"\nAvailable classes:")
                    for i, (obj_class, ids) in enumerate(class_groups.items(), 1):
                        print(f"{i}. {obj_class} ({len(ids)} trajectories): {ids}")
                    
                    class_choice = input(f"\nSelect class (1-{len(class_groups)}): ").strip()
                    try:
                        class_idx = int(class_choice) - 1
                        if 0 <= class_idx < len(class_groups):
                            selected_class = list(class_groups.keys())[class_idx]
                            selected_ids = class_groups[selected_class]
                            print(f"Selected all {selected_class} trajectories: {selected_ids}")
                            return selected_ids
                        else:
                            print("Invalid class selection.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                elif choice == "4":
                    print(f"\nAvailable ID range: {min(sorted_ids)} - {max(sorted_ids)}")
                    try:
                        start_id = int(input("Start ID: ").strip())
                        end_id = int(input("End ID: ").strip())
                        
                        selected_ids = [tid for tid in sorted_ids if start_id <= tid <= end_id]
                        if selected_ids:
                            print(f"Selected IDs {start_id}-{end_id}: {selected_ids}")
                            return selected_ids
                        else:
                            print("No trajectories found in that range.")
                    except ValueError:
                        print("Invalid input. Please enter valid numbers.")
                
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return []
    
    def plot_trajectories(self, output_path: str = None) -> bool:
        """Plot trajectories on satellite image and save result."""
        if not self.sat_image or not self.data:
            print("Error: No data or satellite image loaded")
            return False
        
        trajectories = self.extract_trajectories()
        object_classes = self.get_object_classes()
        
        if not trajectories:
            print("Warning: No trajectories found in the data")
            return False
        
        # Let user select which trajectories to plot
        selected_ids = self.select_trajectory_ids(trajectories, object_classes)
        
        if not selected_ids:
            print("No trajectories selected for plotting.")
            return False
        
        # Filter trajectories to only selected IDs
        selected_trajectories = {tid: trajectories[tid] for tid in selected_ids}
        
        print(f"\nPlotting {len(selected_trajectories)} selected trajectories...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Display satellite image
        ax.imshow(self.sat_image, extent=[0, self.sat_image.width, self.sat_image.height, 0])
        
        # Plot trajectories with unique colors for each ID
        legend_items = []
        for tracked_id in sorted(selected_trajectories.keys()):
            points = selected_trajectories[tracked_id]
            
            if len(points) < 2:
                continue
                
            obj_class = object_classes.get(tracked_id, "unknown")
            color = self.get_color_for_id(tracked_id)  # Unique color per ID
            
            # Extract x and y coordinates
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # Plot trajectory line with thicker line for better visibility
            line, = ax.plot(x_coords, y_coords, color=color, linewidth=3, alpha=0.8)
            
            # Add to legend (limit to avoid clutter)
            if len(legend_items) < 15:
                legend_items.append((line, f"ID {tracked_id} ({obj_class})"))
            
            # Plot start point (large green circle)
            ax.plot(x_coords[0], y_coords[0], 'o', color='lime', markersize=12, 
                   markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
            
            # Plot end point (large red square)
            ax.plot(x_coords[-1], y_coords[-1], 's', color='red', markersize=12, 
                   markeredgecolor='darkred', markeredgewidth=2, alpha=0.9)
        
        # Customize plot
        ax.set_xlim(0, self.sat_image.width)
        ax.set_ylim(self.sat_image.height, 0)  # Invert y-axis to match image coordinates
        ax.set_aspect('equal')
        ax.set_title(f'Selected Object Trajectories - {self.location_name}', fontsize=16, weight='bold')
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        
        # Add legend with better formatting
        if legend_items:
            lines, labels = zip(*legend_items)
            legend = ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1),
                             framealpha=0.9, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
        
        # Add statistics text
        stats_text = f"Selected Trajectories: {len(selected_trajectories)}\n"
        stats_text += f"Total Available: {len(trajectories)}\n"
        stats_text += f"Total Frames: {len(self.data)}\n"
        
        # Count selected classes
        selected_class_counts = {}
        for tid in selected_ids:
            obj_class = object_classes.get(tid, "unknown")
            selected_class_counts[obj_class] = selected_class_counts.get(obj_class, 0) + 1
        
        stats_text += "Selected Classes:\n"
        for obj_class, count in sorted(selected_class_counts.items()):
            stats_text += f"  {obj_class}: {count}\n"
        
        # Add legend for start/end markers
        stats_text += "\nLegend:\n"
        stats_text += "  🟢 Start point\n"
        stats_text += "  🔴 End point"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        if not output_path:
            selected_ids_str = "_".join(map(str, sorted(selected_ids)[:5]))  # First 5 IDs
            if len(selected_ids) > 5:
                selected_ids_str += f"_plus{len(selected_ids)-5}more"
            output_path = f"trajectory_plot_{self.location_name}_IDs_{selected_ids_str}.png"
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {output_path}")
            plt.close()
            return True
            
        except Exception as e:
            print(f"Error: Failed to save plot: {str(e)}")
            return False


def main():
    """Main function to run the trajectory plotter."""
    print("TrafficLab 3D Trajectory Plotter (CLI)")
    print("=" * 40)
    
    plotter = TrajectoryPlotterCLI()
    
    # Get file path from command line argument or interactive selection
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if file_path:
        print(f"Using file from command line: {file_path}")
    else:
        print("Scanning for trajectory files...")
    
    json_file = plotter.select_json_file(file_path)
    
    if not json_file:
        print("No file selected. Exiting.")
        return
    
    print(f"Selected file: {json_file}")
    
    # Load trajectory data
    if not plotter.load_trajectory_data(json_file):
        return
    
    # Load satellite image
    if not plotter.load_satellite_image():
        return
    
    # Plot trajectories
    output_file = f"trajectory_plot_{plotter.location_name}_{Path(json_file).stem.replace('.json', '')}.png"
    
    if plotter.plot_trajectories(output_file):
        print("Trajectory plotting completed successfully!")
        print(f"Output saved to: {output_file}")
    else:
        print("Failed to plot trajectories.")


if __name__ == "__main__":
    main()