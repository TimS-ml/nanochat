"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

def run_command(cmd):
    """
    Run a shell command and return its output, or None if it fails.

    Args:
        cmd: Shell command string to execute

    Returns:
        String output of the command (stripped), or None if command fails or times out

    Note:
        Command will timeout after 5 seconds to prevent hanging.
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def get_git_info():
    """
    Gather git repository information for the training report.

    Returns:
        Dict containing:
        - commit: Short commit hash (7 chars)
        - branch: Current branch name
        - dirty: True if there are uncommitted changes
        - message: First line of commit message (truncated to 80 chars)

    Note:
        Returns "unknown" for fields if git commands fail.
    """
    info = {}
    # Get short commit hash (7 characters)
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    # Get current branch name
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    # git status --porcelain returns empty string for clean repo
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # Get the first line of the latest commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # First line, truncated to 80 chars

    return info

def get_gpu_info():
    """
    Gather GPU information for the training report.

    Returns:
        Dict containing:
        - available: True if CUDA GPUs are available
        - count: Number of GPUs
        - names: List of GPU names (e.g. "NVIDIA A100-SXM4-80GB")
        - memory_gb: List of GPU memory sizes in GB
        - cuda_version: CUDA version string

        If no GPUs are available, returns {"available": False}.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    # Collect information for each GPU
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        # Convert bytes to GB (1024^3)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # Get CUDA version (e.g. "11.7")
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """
    Gather system information for the training report.

    Returns:
        Dict containing:
        - hostname: Machine hostname
        - platform: OS name (Linux, Darwin, Windows)
        - python_version: Python version string
        - torch_version: PyTorch version string
        - cpu_count: Number of physical CPU cores
        - cpu_count_logical: Number of logical CPU cores (with hyperthreading)
        - memory_gb: Total system memory in GB
        - user: Username from environment
        - nanochat_base_dir: Base directory for nanochat outputs
        - working_dir: Current working directory
    """
    info = {}

    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)  # Physical cores
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)  # Logical cores (with hyperthreading)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB

    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """
    Estimate cloud GPU training cost based on GPU type and runtime.

    Uses approximate pricing from Lambda Cloud as a reference.
    The actual cost may vary depending on your cloud provider.

    Args:
        gpu_info: Dict from get_gpu_info() containing GPU details
        runtime_hours: Optional runtime in hours for total cost estimation

    Returns:
        Dict containing:
        - hourly_rate: Estimated cost per hour in USD
        - gpu_type: GPU name used for estimation
        - estimated_total: Total cost estimate (if runtime_hours provided)

        Returns None if no GPUs are available.

    Note:
        Pricing is approximate and based on Lambda Cloud rates:
        - H100: $3.00/hour
        - A100: $1.79/hour
        - V100: $0.55/hour
        - Unknown: $2.00/hour (default)
    """

    # Rough pricing estimates from Lambda Cloud
    default_rate = 2.0  # Default for unknown GPU types
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from the name string
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            # Multiply by number of GPUs for total hourly cost
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        # Unknown GPU type, use default rate
        hourly_rate = default_rate * gpu_info["count"]

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """
    Generate the header section for a training report.

    Creates a markdown-formatted header containing:
    - Timestamp
    - Git information (branch, commit, message)
    - Hardware information (CPUs, GPUs, memory)
    - Software versions (Python, PyTorch, CUDA)
    - Cost estimate (if GPU info available)
    - Codebase bloat metrics (lines, files, dependencies)

    Returns:
        String containing formatted markdown header
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # Bloat metrics: package all source code and measure its size
    # This gives a sense of how complex/bloated the codebase is
    # Uses files-to-prompt tool to gather all relevant source files
    packaged = run_command('files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml')
    num_chars = len(packaged)
    num_lines = len(packaged.split('\n'))
    # Count files by looking for <source> tags in the cxml output
    num_files = len([x for x in packaged.split('\n') if x.startswith('<source>')])
    # Rough estimate: 4 characters per token (typical for English text)
    num_tokens = num_chars // 4

    # Count dependencies from uv.lock file
    # More dependencies = more complexity and potential issues
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r', encoding='utf-8') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """
    Convert text to a URL-friendly slug.

    Args:
        text: String to slugify

    Returns:
        Lowercase string with spaces replaced by hyphens

    Example:
        >>> slugify("Base Model Training")
        'base-model-training'
    """
    return text.lower().replace(" ", "-")

# The expected report section files and their order
# These are generated during the nanochat training pipeline
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "midtraining.md",
    "chat-evaluation-mid.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]

# The evaluation metrics we track for chat models
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """
    Extract metric values from a report section by searching for key strings.

    Args:
        section: String content of a report section
        keys: String or list of strings to search for

    Returns:
        Dict mapping keys to their values (the part after the colon)

    Example:
        >>> section = "ARC-Easy: 0.75\\nGSM8K: 0.42"
        >>> extract(section, ["ARC-Easy", "GSM8K"])
        {'ARC-Easy': '0.75', 'GSM8K': '0.42'}
    """
    if not isinstance(keys, list):
        keys = [keys]  # Allow single string for convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                # Extract the value after the colon
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """
    Extract a timestamp from content that starts with a given prefix.

    Args:
        content: String content to search
        prefix: Line prefix to look for (e.g. "timestamp:")

    Returns:
        datetime.datetime object, or None if not found or invalid format

    Example:
        >>> content = "timestamp: 2025-01-15 10:30:00"
        >>> extract_timestamp(content, "timestamp:")
        datetime.datetime(2025, 1, 15, 10, 30, 0)
    """
    for line in content.split('\n'):
        if line.startswith(prefix):
            # Extract everything after the first colon
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass  # Invalid timestamp format
    return None

class Report:
    """
    Maintains training logs and generates a comprehensive markdown report.

    This class collects metrics and information from various stages of the
    nanochat training pipeline (tokenizer training, base model training,
    finetuning, evaluation, etc.) and assembles them into a final report.

    Attributes:
        report_dir: Directory where report sections and final report are stored
    """

    def __init__(self, report_dir):
        """
        Initialize the report.

        Args:
            report_dir: Directory path for storing report files
        """
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """
        Log a section of data to the report.

        Creates a markdown file for the section with a timestamp and the provided
        data. The data can be a mix of strings and dictionaries.

        Args:
            section: Section name (e.g. "Base Model Training")
            data: List of items to log. Each item can be:
                  - str: Written directly to the file
                  - dict: Rendered as a bullet list of key-value pairs

        Returns:
            str: Path to the created section file

        Example:
            >>> report.log("Training", [
            ...     {"loss": 2.3, "steps": 1000},
            ...     "Additional notes here"
            ... ])
        """
        # Convert section name to filename (e.g. "Base Model Training" -> "base-model-training.md")
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            # Write section header and timestamp
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            # Write data items
            for item in data:
                if not item:
                    # Skip falsy values like None or empty dict
                    continue
                if isinstance(item, str):
                    # Directly write string content
                    f.write(item)
                else:
                    # Render dictionary as bullet list
                    for k, v in item.items():
                        # Format values nicely
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"  # 4 decimal places for floats
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"  # Add commas for large numbers
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """
        Generate the final comprehensive training report.

        Assembles all section files into a single markdown report with:
        - Header with environment and git info
        - All training/evaluation sections in order
        - Summary table with key metrics from each stage
        - Total wall clock time

        The report is saved to both the report directory and copied to the
        current directory for convenience.

        Returns:
            str: Path to the generated report file
        """
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        # Track the most important metrics from each stage for the summary table
        final_metrics = {}
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # Write the header first (contains environment info)
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    # Extract start time for wall clock calculation
                    start_time = extract_timestamp(header_content, "Run started:")
                    # Extract bloat data for the summary section
                    # (the content between "### Bloat\n" and "\n\n")
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None  # Can't calculate wall clock time without start
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")

            # Process all the individual section files in order
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()

                # Extract timestamp from this section
                # The last non-RL section's timestamp becomes the end time
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    # and may not always be run
                    end_time = extract_timestamp(section, "timestamp:")

                # Extract the most important metrics from each section for the summary table
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md":
                    final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K")  # RL only evaluates GSM8K

                # Append this section to the main report
                out_file.write(section)
                out_file.write("\n")
            # Add the summary section with key metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")

            # Build a table showing key metrics across all training stages
            # Collect all unique metric names across all stages
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first (base model metric), ChatCORE last, rest alphabetically
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))

            # Fixed column widths for nice formatting
            stages = ["base", "mid", "sft", "rl"]
            metric_width = 15
            value_width = 8

            # Write markdown table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")

            # Write separator line
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")

            # Write table rows (one per metric)
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    # Get metric value for this stage, or "-" if not available
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")

            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")

        # Copy the report to current directory for convenience
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """
        Reset the report by clearing all section files and creating a fresh header.

        This should be called at the start of a new training run to ensure
        the report starts fresh. It:
        1. Deletes all existing section files
        2. Deletes the previous final report
        3. Generates a new header with current system info and start timestamp
        """
        # Remove all section files from previous run
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove the final report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write a fresh header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

class DummyReport:
    """
    No-op report for non-rank-0 processes in distributed training.

    In distributed training, only rank 0 should write to the report to avoid
    conflicts. Other ranks use this dummy that ignores all logging calls.
    """
    def log(self, *args, **kwargs):
        """No-op log method."""
        pass
    def reset(self, *args, **kwargs):
        """No-op reset method."""
        pass

def get_report():
    """
    Get the appropriate Report instance for the current process.

    In distributed training, only rank 0 gets a real Report instance.
    Other ranks get a DummyReport that ignores all operations.

    Returns:
        Report or DummyReport instance
    """
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        # Only rank 0 writes to the report
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        # Other ranks get a dummy that does nothing
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="Operation to perform (default: generate)")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
