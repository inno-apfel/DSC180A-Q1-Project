def is_2dig_naics(naics_code):
    """
    Checks if a given NAICS code string is a 2-digit NAICS code

    Parameters
    ----------
    naics_code: str
        String representation of an NAICS code
        Expected to range from 2-digit to 5-digit representations 

    Returns
    -------
    boolean:
        Whether or not the input string is a valid 2-digit NAICS code

    Example
    -------
        >>> is_2dig_naics('236///')
        False
        >>> is_2dig_naics('23----')
        True
    """
    return naics_code[:2].isnumeric() and not any(char.isdigit() for char in naics_code[2:])

def reformat_zip(x):
    """
    Reformats ZIP Codes from Census 5-Digit ZIP Code Tabulation Area (ZCTA5) format to simple 5-Digit representation.

    Parameters
    ----------
    x: str
        ZIP code represented in the format 'ZCTA5 XXXXX'

    Returns
    -------
    str:
        ZIP code in 5-digit representation (XXXXX)
    """
    return x[6:11]

def reformat_income(x):
    """
    Reformats and converts to int, string representations of median household incomes

    Parameters
    ----------
    x: str
        String representation of median household income

    Returns
    -------
    str:
        Cleaned integer representation of median household income

    Example
    -------
        >>> reformat_income('250,000+')
        250000
    """
    x = x.replace('+', '')
    return int(x.replace(',', ''))

def reformat_pop(x):
    """
    Reformats and converts to int, string representations of population counts

    Parameters
    ----------
    x: str
        String representation of population count

    Returns
    -------
    str:
        Cleaned integer representation of population count

    Example
    -------
        >>> reformat_pop('1,234')
        1234
    """
    return float(x.replace(',', ''))