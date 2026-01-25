import logging
import sys

def get_logger(name='mowing_terrain_seg', log_level=logging.INFO):
    """
    Initialize and get a logger with a standard format.
    
    Args:
        name (str): The name of the logger.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs if a parent logger exists
        logger.propagate = False
        
    return logger

# Create a default instance for easy access
LOGGER = get_logger()