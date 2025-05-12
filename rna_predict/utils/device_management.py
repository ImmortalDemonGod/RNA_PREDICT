"""
Device management utilities for handling device compatibility issues.

This module provides functions for managing device selection and fallback
behavior in a way that respects Hydra configuration.
"""

import logging
import torch
from typing import Any
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def get_device_for_component(
    cfg: DictConfig, 
    component_path: str, 
    default_device: str = "cpu"
) -> torch.device:
    """
    Get the appropriate device for a component based on configuration.
    
    Args:
        cfg: The Hydra configuration object
        component_path: Dot-separated path to the component (e.g., "model.stageB.torsion_bert")
        default_device: Default device to use if no configuration is found
        
    Returns:
        torch.device: The device to use for this component
    """
    # Check if device management is configured
    if not hasattr(cfg, 'device_management'):
        logger.warning(f"No device_management configuration found, using default device: {default_device}")
        return torch.device(default_device)
    
    # Get primary and fallback devices
    primary_device = getattr(cfg.device_management, 'primary', default_device)
    fallback_device = getattr(cfg.device_management, 'fallback', "cpu")
    auto_fallback = getattr(cfg.device_management, 'auto_fallback', True)
    
    # Check if this component is in the force_to_cpu list
    force_components_to_cpu = getattr(cfg.device_management, 'force_components_to_cpu', [])
    if component_path in force_components_to_cpu:
        logger.info(f"Component {component_path} is configured to always use CPU")
        return torch.device("cpu")
    
    # Check device availability
    if primary_device == "cuda" and not torch.cuda.is_available():
        if auto_fallback:
            logger.warning(f"CUDA requested but not available, falling back to {fallback_device}")
            return torch.device(fallback_device)
        else:
            raise RuntimeError("CUDA requested but not available, and auto_fallback is disabled")
    
    if primary_device == "mps" and not torch.backends.mps.is_available():
        if auto_fallback:
            logger.warning(f"MPS requested but not available, falling back to {fallback_device}")
            return torch.device(fallback_device)
        else:
            raise RuntimeError("MPS requested but not available, and auto_fallback is disabled")
    
    # Return the primary device if it's available
    return torch.device(primary_device)

def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move data to the specified device.
    
    Args:
        data: Data to move (can be a tensor, list, tuple, dict, or other)
        device: Device to move data to
        
    Returns:
        The data moved to the device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data

def handle_device_error(
    component_path: str,
    cfg: DictConfig,
    error: Exception
) -> torch.device:
    """
    Handle device-related errors by falling back to CPU if configured.
    
    Args:
        component_path: Path to the component that encountered the error
        cfg: The Hydra configuration object
        error: The exception that was raised
        
    Returns:
        torch.device: The fallback device to use
        
    Raises:
        The original exception if auto_fallback is disabled
    """
    if not hasattr(cfg, 'device_management'):
        logger.error(f"Device error in {component_path} but no device_management config found")
        raise error
    
    auto_fallback = getattr(cfg.device_management, 'auto_fallback', True)
    fallback_device = getattr(cfg.device_management, 'fallback', "cpu")
    
    if auto_fallback:
        logger.warning(
            f"Device error in {component_path}: {str(error)}. "
            f"Falling back to {fallback_device}"
        )
        # Add this component to the force_to_cpu list for future runs
        if not hasattr(cfg.device_management, 'force_components_to_cpu'):
            cfg.device_management.force_components_to_cpu = []
        if component_path not in cfg.device_management.force_components_to_cpu:
            cfg.device_management.force_components_to_cpu.append(component_path)
        
        return torch.device(fallback_device)
    else:
        logger.error(f"Device error in {component_path} and auto_fallback is disabled")
        raise error
