import re
import subprocess

from counterfactuals2.misc.language import Language


def format_code(code: str, lang: Language | str, mask: str | None = None) -> str:
    if lang == Language.Cpp or lang == "cpp" or lang == "c++":
        return format_cpp(code, mask)
    elif lang == Language.Java or lang == "java" or lang == "Java":
        return format_java(code, mask)
    else:
        print("unknown language", lang, "cannot format")
        return code


def format_cpp(code: str, mask: str | None):
    try:
        code = re.sub(r"#include <(.*?)>\s*", r"#include <\1>\n", code)
        code = re.sub(r"([0-9]+?) \. ([0-9]+?)", r"\1.\2", code)
        code = re.sub(r"([0-9]+?) f", r"\1f", code)
        code = code.replace(" ::", "::")
        code = code.replace("< ", "<")
        code = code.replace(" >", ">")
        # Run clang-format as a subprocess on Windows, passing the code via stdin
        # todo why check = true?
        formatted_code = subprocess.run('clang-format -', input=code, text=True, capture_output=True, check=True,
                                        shell=True).stdout
        if mask is None:
            return formatted_code
        else:
            mask_regex = re.sub(r"(<+)", r"\1\\s*", mask)
            mask_regex = re.sub(r"(>+)", r"\\s*\1", mask_regex)
            return re.sub(mask_regex, mask, formatted_code)

    except subprocess.CalledProcessError as e:
        print("error during code formatting", e)
        return code


def format_java(java_code, mask: str | None):
    if mask is not None:
        return hardcoded_format_java(java_code)
    try:
        # Path to the google-java-format JAR file
        formatter_path = "../Java/google-java-format-1.18.1-all-deps.jar"

        # Run the formatter using subprocess
        result = subprocess.run(['java', '-jar', formatter_path, '-'], input=java_code, text=True,
                                capture_output=True, check=False)
        formatted = result.stdout

        if len(formatted) == 0:
            return hardcoded_format_java(java_code)
        return formatted
    except subprocess.CalledProcessError as e:
        print(f'Error formatting Java code: {e}')
        print(e.stdout, e.stderr)
        return hardcoded_format_java(java_code)


def hardcoded_format_java(java_code):
    hardcoded = java_code.replace("{", " {\n").replace("}", "\n}\n").replace(" . ", ".").replace(
        " ( ", "(").replace(" ) ", ")").replace(" [ ] ", "[] ").replace(" [ ", "[").replace(" ] ", "]")
    hardcoded = re.sub(r"}\s+}", "\n}\n}\n", hardcoded)
    hardcoded = re.sub(r"/\s+/", "//", hardcoded)
    hardcoded = re.sub(r"\s*;\s*", ";\n", hardcoded)
    hardcoded = re.sub(r"\n\s+\n", "\n", hardcoded)
    hardcoded = re.sub(r"  +", " ", hardcoded)
    return hardcoded.strip()
