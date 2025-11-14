"""
Poor Man's Configurator - Simple command-line configuration override system.

This is intentionally simple but unconventional. It allows config overrides via:
1. Config files: python train.py config/my_config.py
2. CLI arguments: python train.py --batch_size=32
3. Both: python train.py config/my_config.py --batch_size=32

How it works:
- This file is exec()'d from training scripts (e.g., train.py)
- Not imported as a module, just runs as inline code
- Directly modifies globals() in the calling script
- Config files are also exec()'d, allowing Python expressions

Pros:
- Very simple, no config. prefix needed
- Full Python expressiveness in config files
- Type checking ensures overrides match defaults

Cons:
- Uses exec() which some consider unsafe/unpythonic
- Not a conventional config system

Example usage:
    $ python train.py config/small_model.py --learning_rate=0.001

If you have a better simple solution, contributions welcome!
"""

import os
import sys
from ast import literal_eval

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

# Parse command-line arguments and apply configuration overrides
for arg in sys.argv[1:]:
    if '=' not in arg:
        # No '=' sign means this is a config file path
        assert not arg.startswith('--'), "File paths should not start with --"
        config_file = arg
        print0(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print0(f.read())
        # Execute the config file, which sets variables in globals()
        exec(open(config_file).read())
    else:
        # Has '=' sign means this is a --key=value override
        assert arg.startswith('--'), "Key-value overrides must start with --"
        key, val = arg.split('=')
        key = key[2:]  # Strip the '--' prefix
        if key in globals():
            try:
                # Try to parse as a Python literal (int, float, bool, list, etc.)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # If parsing fails, treat as a string
                attempt = val
            # Type safety: ensure override matches the type of the default value
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"Type mismatch for {key}: {attempt_type} != {default_type}"
            # Apply the override
            print0(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
