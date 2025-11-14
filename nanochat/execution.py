"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
Adapted from OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

What is covered:
- Each execution runs in its own process (can be killed if it hangs or crashes)
- Execution is limited by a timeout to stop infinite loops
- Memory limits are enforced by default (256MB)
- stdout and stderr are captured and returned
- Code runs in a temporary directory that is deleted afterwards
- Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

What is not covered:
- Not a true security sandbox
- Network access is not blocked (e.g. sockets could be opened)
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- No kernel-level isolation (no seccomp, no containers, no virtualization)

Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Result of executing Python code in a sandbox.

    Attributes:
        success: True if code executed without errors, False otherwise
        stdout: Captured standard output from the code execution
        stderr: Captured standard error from the code execution
        error: Error message if execution failed (None if successful)
        timeout: True if execution exceeded time limit
        memory_exceeded: True if execution exceeded memory limit
    """
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False

    def __repr__(self):
        """
        Custom string representation that only shows non-empty/non-default fields.
        Makes the output more readable by omitting unnecessary information.
        """
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Context manager to enforce a time limit on code execution using signals.

    Args:
        seconds: Maximum execution time in seconds (can be fractional)

    Raises:
        TimeoutException: If execution time exceeds the limit

    Note:
        Uses SIGALRM which only works on Unix-like systems (not Windows).
        The timer is automatically cleared on exit even if an exception occurs.
    """
    def signal_handler(signum, frame):
        """Signal handler that raises TimeoutException when alarm fires."""
        raise TimeoutException("Timed out!")

    # Set up a real-time interval timer that fires after 'seconds'
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        # Clear the timer to prevent it from firing after we exit the context
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def capture_io():
    """
    Capture stdout and stderr, and disable stdin for untrusted code execution.

    This prevents code from:
    - Reading from stdin (raises IOError)
    - Printing to the terminal directly (captured instead)

    Yields:
        Tuple of (stdout_capture, stderr_capture): StringIO objects containing output

    Example:
        with capture_io() as (stdout, stderr):
            print("hello")  # Captured, not printed to terminal
            input()  # Raises IOError
        output = stdout.getvalue()  # "hello\n"
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()  # Prevents reading from stdin
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    """
    Create a temporary directory and change to it for the duration of the context.

    The directory is automatically deleted when exiting the context, along with
    all files created inside it. This provides isolation for file operations.

    Yields:
        str: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    """Exception raised when code execution exceeds the time limit."""
    pass


class WriteOnlyStringIO(io.StringIO):
    """
    StringIO subclass that raises IOError on any read operation.

    This is used to block stdin access in untrusted code execution,
    preventing code from waiting for user input (which would hang).
    """

    def read(self, *args, **kwargs):
        """Raise IOError instead of reading."""
        raise IOError

    def readline(self, *args, **kwargs):
        """Raise IOError instead of reading a line."""
        raise IOError

    def readlines(self, *args, **kwargs):
        """Raise IOError instead of reading all lines."""
        raise IOError

    def readable(self, *args, **kwargs):
        """
        Returns False to indicate this stream cannot be read from.

        Returns:
            bool: Always False
        """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    """
    Context manager to redirect stdin, similar to redirect_stdout/redirect_stderr.

    This is not provided by the standard library but follows the same pattern.
    """
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    """
    Temporarily change to a different directory.

    Args:
        root: Directory to change to ("." to stay in current directory)

    Note:
        The original directory is restored even if an exception occurs.
    """
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        # Always restore the original directory
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    Disable destructive functions to protect against accidental or malicious code.

    This function neuters potentially dangerous operations like:
    - File system modifications (remove, rename, chmod, etc.)
    - Process management (fork, kill, system, subprocess)
    - Resource manipulation (setuid, chroot, etc.)
    - Memory-intensive operations (enforces memory limits)

    Args:
        maximum_memory_bytes: Maximum memory usage in bytes (None to skip)

    WARNING:
        This function is NOT a security sandbox. Untrusted code, including model-
        generated code, should not be blindly executed outside of a proper sandbox.
        See the Codex paper for more information about OpenAI's code sandbox.

    Note:
        - Memory limits only work on non-macOS systems
        - This provides defense-in-depth against accidental issues, not security
        - Sophisticated attackers could still bypass these restrictions
    """

    # Set memory limits (doesn't work reliably on macOS)
    if platform.uname().system != "Darwin":
        import resource
        # RLIMIT_AS: maximum area (in bytes) of address space
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        # RLIMIT_DATA: maximum size of the process's data segment
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        # RLIMIT_STACK: maximum size of the process stack
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    # Disable fault handler to prevent it from interfering with our error handling
    faulthandler.disable()

    # Disable built-in exit functions
    import builtins
    builtins.exit = None
    builtins.quit = None

    import os

    # Limit OpenMP to single thread to reduce resource usage
    os.environ["OMP_NUM_THREADS"] = "1"

    # Disable process management functions (prevent fork bombs, process killing)
    os.kill = None
    os.system = None
    os.putenv = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.setuid = None

    # Disable file system modification functions (prevent file deletion/modification)
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None

    # Disable file permission modification functions
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.lchmod = None
    os.lchown = None
    os.lchflags = None

    # Disable directory navigation functions
    os.fchdir = None
    os.chroot = None
    os.chdir = None
    os.getcwd = None

    # Disable shutil destructive operations
    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    # Disable subprocess to prevent shell command execution
    import subprocess
    subprocess.Popen = None  # type: ignore

    # Disable help function (can leak information about the system)
    __builtins__["help"] = None

    # Block potentially dangerous modules from being imported
    import sys
    sys.modules["ipdb"] = None  # debugger
    sys.modules["joblib"] = None  # parallel processing
    sys.modules["resource"] = None  # resource limits manipulation
    sys.modules["psutil"] = None  # system/process utilities
    sys.modules["tkinter"] = None  # GUI (can hang)


def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    """
    Execute code in a subprocess with safety guards. Internal function.

    This function runs in a separate process and communicates results back
    through a shared dictionary. It applies multiple layers of protection:
    1. Runs in a temporary directory (isolated file system)
    2. Applies reliability guard (disables dangerous functions)
    3. Captures stdout/stderr (prevents terminal output)
    4. Enforces time limit (prevents infinite loops)
    5. Enforces memory limit (prevents memory bombs)

    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds
        maximum_memory_bytes: Memory limit in bytes (None to skip)
        result_dict: Shared dictionary to write results to

    Note:
        This function is designed to run in a separate process created by
        multiprocessing. It should not be called directly.
    """
    with create_tempdir():

        # Save references to system calls needed for cleanup
        # We need to do this BEFORE calling reliability_guard which neuters them
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Disable destructive functions that could harm the system
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # Initialize result dict with failure state
        # If anything goes wrong, we want to report failure by default
        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            # Create a fresh namespace for code execution
            # This prevents code from accessing local variables in this function
            exec_globals = {}
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(code, exec_globals)

            # Success! Update result dict with captured output
            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })

        except TimeoutException:
            # Code took too long to execute
            result_dict.update({
                "timeout": True,
                "error": "Execution timed out",
            })

        except MemoryError as e:
            # Code exceeded memory limit
            result_dict.update({
                "memory_exceeded": True,
                "error": f"Memory limit exceeded: {e}",
            })

        except BaseException as e:
            # Any other error (SyntaxError, RuntimeError, etc.)
            result_dict.update({
                "error": f"{type(e).__name__}: {e}",
            })

        # Restore system calls needed for cleanup
        # The tempdir context manager needs these to clean up after itself
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0, # 5 seconds default
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024, # 256MB default
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment with safety constraints.

    This is the main entry point for code execution. It provides:
    - Process isolation: Code runs in a separate process
    - Time limits: Prevents infinite loops
    - Memory limits: Prevents memory bombs (on non-macOS systems)
    - I/O capture: Captures stdout/stderr
    - Filesystem isolation: Code runs in a temporary directory
    - Disabled dangerous functions: Prevents destructive operations

    The code is executed via exec() in a fresh namespace, with various
    safety guards applied. If the process hangs or crashes, it will be
    forcefully terminated.

    Args:
        code: Python code to execute as a string
        timeout: Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: Memory limit in bytes (default: 256MB, None to disable)

    Returns:
        ExecutionResult with success status, stdout/stderr, and error information

    Example:
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'

        >>> result = execute_code("1/0")
        >>> result.success
        False
        >>> result.error
        'ZeroDivisionError: division by zero'

    Note:
        - This is NOT a security sandbox, only defense-in-depth
        - Do not execute untrusted code outside a proper sandbox
        - See module docstring for details on what is/isn't protected
    """

    # Create a manager for inter-process communication
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # Start code execution in a separate process for isolation
    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()
    # Wait for the process to complete, with a grace period of 1 second
    # beyond the timeout (to allow the internal timeout handler to work)
    p.join(timeout=timeout + 1)

    # If the process is still alive after timeout + 1 second, forcefully kill it
    if p.is_alive():
        p.kill()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",
            timeout=True,
            memory_exceeded=False,
        )

    # If the process returned no results, something went seriously wrong
    if not result_dict:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution failed (no result returned)",
            timeout=True,
            memory_exceeded=False,
        )

    # Return the execution result from the subprocess
    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )

