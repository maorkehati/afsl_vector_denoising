def validate_config(config):
    """
    Validates that all required keys exist in the config dictionary.

    Args:
        config (dict): Parsed YAML config
    Raises:
        ValueError: If any required field is missing.
    """
    required_top_keys = ['experiment_name', 'model', 'data', 'eval']
    required_model_keys = ['type']
    required_data_keys = ['type', 'path']
    required_eval_keys = ['type', 'test_path', 'metrics']
    
    if config is None or not isinstance(config, dict):
        raise ValueError("Configuration file is empty or not a valid dictionary.")

    # Top-level keys
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level config key: '{key}'")

    # Model
    for key in required_model_keys:
        if key not in config['model']:
            raise ValueError(f"Missing required key in 'model': '{key}'")

    # Data
    for key in required_data_keys:
        if key not in config['data']:
            raise ValueError(f"Missing required key in 'data': '{key}'")

    # Eval
    for key in required_eval_keys:
        if key not in config['eval']:
            raise ValueError(f"Missing required key in 'eval': '{key}'")
