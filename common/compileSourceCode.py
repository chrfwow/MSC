import clang.cindex
import subprocess

from counterfactuals2.misc.language import Language

clang.cindex.Config.set_library_file('D:/Programme/LLVM/bin/libclang.dll')
index = clang.cindex.Index.create()


def is_syntactically_correct(code: str, lang: str | Language):
    if lang == Language.Cpp or lang == "cpp" or lang == "c++":
        return is_syntactically_correct_cpp_code(code)
    elif lang == Language.Java or lang == "java" or lang == "Java":
        return is_syntactically_correct_java_code(code)
    else:
        print("unknown language", lang, "cannot compile")
        return code


def is_syntactically_correct_cpp_code(code) -> bool:
    # Parse the C++ code
    tu = index.parse("tmp.cpp", args=[
        "'C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\Llvm\\bin'"],
                     unsaved_files=[("tmp.cpp", code)])

    for d in tu.diagnostics:
        if d.severity == 4:
            return False
        if d.severity == 3:
            return False
    return True


def is_syntactically_correct_java_code(java_code) -> bool:
    try:
        # Path to the google-java-format JAR file
        formatter_path = "../Java/google-java-format-1.18.1-all-deps.jar"

        # Run the formatter using subprocess
        result = subprocess.run(['java', '-jar', formatter_path, '-'], input=java_code, text=True,
                                capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False
