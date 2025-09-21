"""
Logging Usage Examples

This script demonstrates how to use the logging system in the CrewAI Investment Analysis System.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logging_config import setup_logger, setup_detailed_logger

def demonstrate_logging():
    """Demonstrate the logging system usage"""
    
    # Set up loggers
    logger = setup_logger('demo')
    detailed_logger = setup_detailed_logger('demo')
    
    # Log messages at different levels
    logger.info("This is an INFO message - visible in console and general log file")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Detailed logger messages
    detailed_logger.debug("This is a DEBUG message - only in debug log files")
    detailed_logger.info("This is an INFO message with detailed logger")
    
    # Log with exception information
    try:
        raise ValueError("Example exception for demonstration")
    except ValueError as e:
        detailed_logger.error("Caught an exception", exc_info=True)
    
    print("Logging demonstration completed. Check the logs directory for output files.")

if __name__ == "__main__":
    demonstrate_logging()