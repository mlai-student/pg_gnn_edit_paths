import pkg_resources
import subprocess
import re

def get_installed_packages():
    """Get a dictionary of installed packages and their versions."""
    installed = {}
    for package in pkg_resources.working_set:
        installed[package.key] = package.version

    # Handle special cases with package naming
    if 'torch-geometric' in installed:
        installed['torch_geometric'] = installed['torch-geometric']

    return installed

def get_required_packages():
    """Parse requirements.txt and return a list of required packages and check for CUDA support."""
    required = []
    requires_cuda = False
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Check for CUDA support in separate lines
            if (line.startswith('--find-links') or line.startswith('--index-url')) and 'pytorch.org/whl/cu' in line:
                requires_cuda = True
                continue
            # Skip empty lines, comments, and special options that are on their own line
            if not line or line.startswith('#') or (line.startswith('--') and ' ' not in line):
                continue

            # Check for package with options on the same line
            if ' --' in line:
                # Split the line at the first occurrence of ' --'
                parts = line.split(' --', 1)
                package_part = parts[0].strip()
                options_part = '--' + parts[1].strip()

                # Check for CUDA support in options
                if 'pytorch.org/whl/cu' in options_part:
                    requires_cuda = True

                # Extract package name (without version specifiers)
                package = re.split('[<>=~]', package_part)[0].strip()
            else:
                # Extract package name (without version specifiers) for normal lines
                package = re.split('[<>=~]', line)[0].strip()

            required.append(package)
    return required, requires_cuda

def check_torch_cuda():
    """Check if PyTorch has CUDA support."""
    try:
        result = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            check=True
        )
        has_cuda = result.stdout.strip() == "True"
        return has_cuda
    except subprocess.CalledProcessError:
        return False

def main():
    installed_packages = get_installed_packages()
    required_packages, requires_cuda = get_required_packages()

    print("Checking packages in virtual environment against requirements.txt...")
    print("\nRequired packages:")

    missing_packages = []
    for package in required_packages:
        package_lower = package.lower()
        if package_lower in installed_packages:
            print(f"✓ {package} (version: {installed_packages[package_lower]})")
        else:
            print(f"✗ {package} (MISSING)")
            missing_packages.append(package)

    # Special check for PyTorch CUDA support
    if 'torch' in installed_packages:
        has_cuda = check_torch_cuda()
        if has_cuda:
            print("\n✓ PyTorch has CUDA support")
        else:
            print("\n✗ PyTorch does NOT have CUDA support (CPU only)")
            if requires_cuda:
                print("  requirements.txt specifies CUDA support via: --find-links https://download.pytorch.org/whl/cu118")

    print("\nSummary:")
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
    else:
        print("All required packages are installed.")

    if 'torch' in installed_packages:
        if requires_cuda and not has_cuda:
            print("PyTorch is installed but without CUDA support (required by requirements.txt).")
        elif not requires_cuda and has_cuda:
            print("PyTorch is installed with CUDA support (not required by requirements.txt).")
        elif not requires_cuda and not has_cuda:
            print("PyTorch CPU version is installed as specified in requirements.txt.")

if __name__ == "__main__":
    main()
