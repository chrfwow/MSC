test = """
this is some test code:
    and some indent
    
    some more

some less"""


def to_diff(code: str, file_name: str = "file.c") -> str:
    code_lines = code.strip().split("\n")
    code = "\n +".join(code_lines)
    code = " +" + code
    return """
diff --combined {fileName}
index fabadb8,fabadb9
--- {fileName}
+++ {fileName}
@@@ -1,0 +1,{numLines} @@@
{code}""".format(fileName=file_name, numLines=len(code_lines), code=code).lstrip()


def to_diff_hunk(code: str):
    code_lines = code.strip().split("\n")
    code = "\n +".join(code_lines)
    code = " +" + code
    return """
@@@ -1,0 +1,{numLines} @@@
{code}""".format(numLines=len(code_lines), code=code).lstrip()
