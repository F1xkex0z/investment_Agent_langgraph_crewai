"""
Logging Configuration for CrewAI Investment System

This module provides centralized logging configuration for the entire system.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.DEBUG
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = 'logs/crewai_investment_system.log'

def setup_logger(
    name: str, 
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_format: Log message format
        log_file: Optional log file path (uses default if None)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console gets INFO and above
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)  # File gets all levels based on config
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_detailed_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None,
    debug_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a detailed logger with separate debug and general log files.
    
    Args:
        name: Logger name
        level: Logging level
        log_format: Log message format
        log_file: General log file path
        debug_file: Detailed debug log file path
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(f"detailed_{name}")
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create general file handler
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create debug file handler
    if debug_file is None:
        debug_file = f"logs/debug_{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Ensure debug logs directory exists
    debug_dir = os.path.dirname(debug_file)
    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    debug_handler = logging.FileHandler(debug_file, encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Global logger for the system
system_logger = setup_logger('crewai_system')