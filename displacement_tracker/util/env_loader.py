import os
from functools import wraps
from dotenv import load_dotenv
import yaml
import re


class EnvFileNotFoundError(FileNotFoundError):
    pass


class EnvKeyMissingError(KeyError):
    pass


def require_env_file(required_keys=None):
    """
    Decorator to ensure a .env file exists and contains required keys before running the function.
    Usage:
        @require_env_file
        def foo(...): ...

        @require_env_file(['KEY1', 'KEY2'])
        def bar(...): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Locate .env file in current or parent directories
            current_dir = os.getcwd()
            env_path = None
            while True:
                candidate = os.path.join(current_dir, ".env")
                if os.path.isfile(candidate):
                    env_path = candidate
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    raise EnvFileNotFoundError(
                        ".env file not found in current or parent directories."
                    )
                current_dir = parent_dir
            # Parse .env file
            load_dotenv(env_path)
            # Validate required keys
            if required_keys:
                missing = [k for k in required_keys if not os.getenv(k)]
                if missing:
                    raise EnvKeyMissingError(
                        f"Missing required keys in env: {', '.join(missing)}"
                    )
            return func(*args, **kwargs)

        return wrapper

    # Support both @require_env_file and @require_env_file([...])
    if callable(required_keys):
        return decorator(required_keys)
    return decorator


def load_yaml_with_env(yaml_path):
    """
    Load a YAML file and substitute environment variables marked as ${VAR_NAME}.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Parsed YAML content as a dictionary

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        KeyError: If an environment variable is referenced but not defined
    """
    load_dotenv()

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        content = f.read()

    # Replace ${VAR_NAME} with environment variable values
    def replace_env_var(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            raise KeyError(f"Environment variable '{var_name}' not found")
        return value

    content = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_env_var, content)

    return yaml.safe_load(content)
