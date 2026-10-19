r"""
Package the Road Monitoring System project into a clean, distributable ZIP archive.
Excludes venv, __pycache__, .git, and bulky temporary archives.
"""
import os
import sys
import zipfile
import time

def create_project_zip(output_zip_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.abspath(os.path.join(base_dir, "..", "..")) # C:\Users\Student\Downloads
    
    if output_zip_path is None:
        output_zip_path = os.path.join(downloads_dir, "road_monitoring_system_complete.zip")

    # Exclude patterns
    excluded_dirs = {
        "venv", 
        "__pycache__", 
        ".git", 
        ".pytest_cache", 
        ".idea", 
        ".vscode",
        ".system_generated"
    }
    
    excluded_files = {
        "annotated_phase3.mp4", # ~198MB redundant earlier test
        "phase3_update.zip",
        os.path.basename(output_zip_path)
    }
    
    excluded_extensions = {".pyc", ".pyo"}

    print(f"[*] Packaging project from: {base_dir}")
    print(f"[*] Target ZIP destination: {output_zip_path}")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    start_time = time.time()
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]

            for file in files:
                if file in excluded_files:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in excluded_extensions:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                
                # Add under top-level 'road_monitoring_system/' folder in zip
                arc_name = os.path.join("road_monitoring_system", rel_path)

                try:
                    file_size = os.path.getsize(full_path)
                    zipf.write(full_path, arc_name)
                    total_files += 1
                    total_uncompressed_bytes += file_size
                except Exception as e:
                    print(f"[-] Warning: Failed to add {rel_path}: {e}")

    duration = time.time() - start_time
    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    uncomp_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print("      PROJECT PACKAGING COMPLETE")
    print("=" * 60)
    print(f"  Total Files Included:    {total_files}")
    print(f"  Uncompressed Size:       {uncomp_size_mb:.2f} MB")
    print(f"  Compressed ZIP Size:     {zip_size_mb:.2f} MB")
    print(f"  Compression Ratio:       {(1 - (zip_size_mb / uncomp_size_mb)) * 100:.1f}%")
    print(f"  Elapsed Time:            {duration:.2f} seconds")
    print(f"  ZIP File Path:           {output_zip_path}")
    print("=" * 60 + "\n")
    
    return output_zip_path

if __name__ == "__main__":
    create_project_zip()
