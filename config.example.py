"""
Configuration template for GroundedPRM.

Copy this file to config.py and update with your actual values.
"""

# API Configuration
QWEN_API_URL = "http://qwen2.5-7b-instruct:8001/v1/chat/completions"
QWEN_MODEL_NAME = "qwen2.5-7b-instruct"

DEEPSEEK_API_URL = "http://deepseek-distill-32b:8000/v1/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-distill-32b"

# OpenAI API Key (for optional OpenAI API usage)
OPENAI_API_KEY = ""  # Set via environment variable or .env file

# Concurrency Settings
SEMAPHORE_LIMIT = 40  # Max concurrent async requests

# MCTS Parameters
DEFAULT_EXPLORATION_WEIGHT = 1.414
DEFAULT_CHILD_NUMS = 3
DEFAULT_SIMULATION_DEPTH = 7
DEFAULT_EXECUTE_ROUND = 8

# Data Generation Settings
REQUIRED_POSITIVE_SAMPLES = 3
REQUIRED_NEGATIVE_SAMPLES = 3

# Wolfram Alpha Settings
WA_QUERY_RETRIES = 5

# Math Verification Settings
MATH_VERIFY_TOLERANCE = 0.1  # Absolute error tolerance for numerical comparison
