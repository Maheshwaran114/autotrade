#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Date utilities for handling KiteConnect API date formats.
Provides robust date parsing and conversion functions.
"""

import logging
from datetime import datetime, date
from typing import Union, Optional

logger = logging.getLogger(__name__)


def normalize_expiry_date(expiry_input: Union[str, date, datetime, None]) -> Optional[str]:
    """
    Normalize expiry date from various formats to standard YYYY-MM-DD string.
    
    Args:
        expiry_input: Date in various formats (string, date, datetime, or None)
        
    Returns:
        str: Normalized date string in YYYY-MM-DD format, or None if invalid
    """
    if expiry_input is None:
        return None
    
    try:
        if isinstance(expiry_input, date):
            # Handle both datetime.date and datetime.datetime
            return expiry_input.strftime("%Y-%m-%d")
        elif isinstance(expiry_input, str):
            # Try to parse string and return normalized format
            if not expiry_input.strip():
                return None
            # Parse and reformat to ensure consistency
            parsed_date = datetime.strptime(expiry_input.strip(), "%Y-%m-%d").date()
            return parsed_date.strftime("%Y-%m-%d")
        else:
            logger.warning(f"Unexpected expiry format: {expiry_input} (type: {type(expiry_input)})")
            return None
            
    except ValueError as e:
        logger.warning(f"Could not parse expiry date '{expiry_input}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing expiry date '{expiry_input}': {e}")
        return None


def parse_expiry_to_date(expiry_input: Union[str, date, datetime, None]) -> Optional[date]:
    """
    Parse expiry date from various formats to datetime.date object.
    
    Args:
        expiry_input: Date in various formats
        
    Returns:
        date: Parsed date object, or None if invalid
    """
    if expiry_input is None:
        return None
    
    try:
        if isinstance(expiry_input, date):
            # If it's already a date, return as-is (handles both date and datetime)
            return expiry_input if isinstance(expiry_input, date) else expiry_input.date()
        elif isinstance(expiry_input, str):
            return datetime.strptime(expiry_input.strip(), "%Y-%m-%d").date()
        else:
            logger.warning(f"Cannot parse expiry to date: {expiry_input} (type: {type(expiry_input)})")
            return None
            
    except ValueError as e:
        logger.warning(f"Could not parse expiry date '{expiry_input}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing expiry date '{expiry_input}': {e}")
        return None


def is_valid_expiry_date(expiry_input: Union[str, date, datetime, None]) -> bool:
    """
    Check if the given input is a valid expiry date.
    
    Args:
        expiry_input: Date in various formats
        
    Returns:
        bool: True if valid, False otherwise
    """
    return normalize_expiry_date(expiry_input) is not None


def days_until_expiry(expiry_input: Union[str, date, datetime], reference_date: date) -> Optional[int]:
    """
    Calculate days between reference date and expiry date.
    
    Args:
        expiry_input: Expiry date in various formats
        reference_date: Reference date for calculation
        
    Returns:
        int: Number of days until expiry (positive = future, negative = past)
        None: If dates cannot be parsed
    """
    expiry_date = parse_expiry_to_date(expiry_input)
    if expiry_date is None:
        return None
    
    return (expiry_date - reference_date).days


def filter_valid_expiries(expiry_list: list, reference_date: date, max_days_ahead: int = 28) -> list:
    """
    Filter a list of expiry dates to only include valid, future dates within range.
    
    Args:
        expiry_list: List of expiry dates in various formats
        reference_date: Reference date for filtering
        max_days_ahead: Maximum days ahead to consider
        
    Returns:
        list: Filtered list of normalized expiry date strings
    """
    valid_expiries = []
    
    for expiry in expiry_list:
        normalized_expiry = normalize_expiry_date(expiry)
        if normalized_expiry is None:
            continue
            
        days_diff = days_until_expiry(normalized_expiry, reference_date)
        if days_diff is not None and 0 <= days_diff <= max_days_ahead:
            valid_expiries.append(normalized_expiry)
    
    return sorted(valid_expiries)


def format_expiry_for_symbol(expiry_input: Union[str, date, datetime]) -> Optional[str]:
    """
    Format expiry date for use in trading symbols (e.g., 25MAY format).
    
    Args:
        expiry_input: Expiry date in various formats
        
    Returns:
        str: Formatted expiry for symbol (e.g., "25MAY"), or None if invalid
    """
    expiry_date = parse_expiry_to_date(expiry_input)
    if expiry_date is None:
        return None
    
    return expiry_date.strftime("%d%b").upper()


if __name__ == "__main__":
    # Test the utility functions
    from datetime import date, datetime
    
    print("Testing date utility functions:")
    test_dates = [
        "2025-05-29",
        date(2025, 5, 29),
        datetime(2025, 5, 29, 15, 30),
        None,
        "",
        "invalid-date",
        date(2025, 6, 26),
    ]
    
    reference = date(2025, 5, 24)
    
    for test_date in test_dates:
        normalized = normalize_expiry_date(test_date)
        parsed = parse_expiry_to_date(test_date)
        valid = is_valid_expiry_date(test_date)
        days = days_until_expiry(test_date, reference) if valid else None
        symbol_format = format_expiry_for_symbol(test_date) if valid else None
        
        print(f"Input: {test_date} ({type(test_date).__name__})")
        print(f"  Normalized: {normalized}")
        print(f"  Parsed: {parsed}")
        print(f"  Valid: {valid}")
        print(f"  Days until: {days}")
        print(f"  Symbol format: {symbol_format}")
        print()
    
    print("✅ Date utilities test completed")
