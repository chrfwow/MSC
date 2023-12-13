import subprocess
import re


def format_code(code: str, lang: str) -> str:
    if lang == "cpp" or lang == "c++":
        return format_cpp(code)
    elif lang == "java" or lang == "Java":
        return format_java(code)
    else:
        print("unknown language", lang, "cannot format")
        return code


def format_cpp(code: str):
    try:
        code = re.sub(r"#include <(.*?)>\s*", r"#include <\1>\n", code)
        code = re.sub(r"([0-9]+?) \. ([0-9]+?)", r"\1.\2", code)
        code = re.sub(r"([0-9]+?) f", r"\1f", code)
        code = code.replace(" ::", "::")
        code = code.replace("< ", "<")
        code = code.replace(" >", ">")
        # Run clang-format as a subprocess on Windows, passing the code via stdin
        formatted_code = subprocess.run('clang-format -', input=code, text=True, capture_output=True, check=True,
                                        shell=True).stdout
        return formatted_code
    except subprocess.CalledProcessError as e:
        print("error during code formatting", e)
        return code


def format_java(java_code):
    try:
        # Path to the google-java-format JAR file
        formatter_path = "../Java/google-java-format-1.18.1-all-deps.jar"

        # Run the formatter using subprocess
        result = subprocess.run(['java', '-jar', formatter_path, '-'], input=java_code, text=True,
                                capture_output=True, check=False)
        formatted = result.stdout
        if len(formatted) == 0:
            return java_code
        return formatted
    except subprocess.CalledProcessError as e:
        print(f'Error formatting Java code: {e}')
        print(e.stdout, e.stderr)
        return java_code