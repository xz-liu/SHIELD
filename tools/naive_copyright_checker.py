import os
import re
from pathlib import Path
from datetime import datetime

def is_public_domain(region: str, cur_year: int, pub_year: int, death_year: int = None) -> bool:
    """death_year is None means the author has not died. cur_year >= 2024."""
    
    if pub_year < 1923:
        return True

    if region == "US": 
        # Duration of Copyright: https://www.copyright.gov/title17/92chap3.html
        if pub_year < 1978:
            return cur_year - pub_year > 95 # if the copyright is renewed, its max duration is 95 years
        return death_year is not None and cur_year - death_year > 70

    if region == "CA":
        # reference: https://ised-isde.canada.ca/site/canadian-intellectual-property-office/en/what-intellectual-property/what-copyright
        if death_year is not None:
            if cur_year - death_year <= 70:
                return True  
            if death_year < 2022 - 50:
                # before 2022, the copyright protection ended after death year plus 50
                # for works in public domain, it will be no longer protected even 
                # the copyright duration is extended later
                return True  
        return False

    if region in {"EU", "KR", "AUS", "RU"}:
        return death_year is not None and cur_year - death_year > 70
    
    if region == "CN":
        return death_year is not None and cur_year - death_year > 50

    if region == "IN":
        return death_year is not None and cur_year - death_year > 60

    if region == "JP":
        if pub_year < 1926:
            return True
        return death_year is not None and cur_year - death_year > 70

    return False


def test_is_public_domain():
    assert is_public_domain('US', 2024, 1920) == True, "Test case 1 failed"
    assert is_public_domain('US', 2024, 1975) == False, "Test case 2 failed"
    assert is_public_domain('US', 2024, 1950, 1980) == False, "Test case 3 failed"
    assert is_public_domain('US', 2024, 1980) == False, "Test case 4 failed"
    assert is_public_domain('US', 2024, 1977, 1940) == False, "Test case 5 failed"
    assert is_public_domain('US', 2024, 1960) == False, "Test case 6 failed"

    print("All test cases passed")
    
    
def check_is_pub(path: Path, region: str, is_pub: bool):
    cur_year = datetime.now().year
    for file in path.glob("*.txt"):
        death_year = re.search(r'\(d\. (\d{4})\)', file.name)
        pub_year = re.search(r'\(p\. (\d{4})\)', file.name)
        death_year = int(death_year.group(1)) if death_year else None
        pub_year = int(pub_year.group(1))
        if is_pub != is_public_domain(region, cur_year, pub_year, death_year):
            print(file.name)

def check_bsc():
    for ct in countries:
        check_is_pub(Path(os.path.expanduser('../datasets/bsc')), ct, False)

def check_bsnc():
    for ct in countries:
        check_is_pub(Path(os.path.expanduser('../datasets/bsnc')), ct, True)

def check_bsmc():
    check_is_pub(Path(os.path.expanduser('../datasets/bsmc')), "US", False)
    check_is_pub(Path(os.path.expanduser('../datasets/bsmc')), "CN", True)


if __name__ == "__main__":
    countries = ["CN", "US", "CA", "JP", "EU", "KR", "AUS", "CA", "RU"]
    check_bsc()
    check_bsnc()
    check_bsmc()

    test_is_public_domain()
