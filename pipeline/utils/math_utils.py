from math_verify import parse, verify
import re
import logging

logger = logging.getLogger(__name__)


def last_boxed_only_string(string):
    if "\\boxed{" in string:
        pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, string)
        if match:
            answer = match[-1] # Return the content inside \boxed{}
            return answer.strip('.')
        else:
            pattern2 = r"\$([^$]*)\$"
            match2 = re.findall(pattern2, string)
            if match2:
                # match the last content in $$
                answer = match2[-1]
                return answer.strip('.')
            else:
                return "Cannot find the boxed final result in the answer"
    elif "oxed{" in string:
        pattern = r"oxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        match = re.findall(pattern, string)
        if match:
            answer = match[-1] # Return the content inside \boxed{}
            return answer.strip('.')
        else:
            pattern2 = r"\$([^$]*)\$"
            match2 = re.findall(pattern2, string)
            if match2:
                # match the last content in $$
                answer = match2[-1]
                return answer.strip('.')
            else:
                return "Cannot find the boxed final result in the answer"
    else:
        return string


def parse_math_boxed(s):
    if not s:
        return "N/A"
    s = last_boxed_only_string(s)
    return s


def is_math_correct(gold, answer) -> bool:
    try:
        gold = parse("$"+gold+"$")  
        answer = parse("$"+answer+"$") 
    except Exception as e:
        logger.warning(f"Error parsing math: {e}, gold={gold}, answer={answer}")
    try:
        output = verify(gold, answer) 
        return output
    except BaseException as e:
        logger.warning(f"Error comparing math: {e}, gold={gold}, answer={answer}")
        return False
