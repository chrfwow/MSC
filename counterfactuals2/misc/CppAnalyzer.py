import clang.cindex

idx = clang.cindex.Index.create()

keywords = {
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "atomic_cancel",
    "atomic_commit",
    "atomic_noexcept",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char8_t",
    "char16_t",
    "char32_t",
    "int64_t",
    "int32_t",
    "int16_t",
    "int8_t",
    "class",
    "compl",
    "concept",
    "const",
    "consteval",
    "constexpr",
    "constinit",
    "const_cast",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "reflexpr",
    "register",
    "reinterpret_cast",
    "requires",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "synchronized",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
}

loop_keywords = {
    "for",
    "while",
    "do",
}


def analyze_keyword(tokens):
    for t in tokens:
        s = t.spelling
        if t.kind != clang.cindex.TokenKind.KEYWORD and t.spelling not in keywords:
            pass
        elif s in keywords:
            if s in loop_keywords:
                return "LOOP:" + s
            else:
                return "KEYWORD:" + s


def analyze_punctation(tokens):
    for t in tokens:
        s = t.spelling
        if t.kind == clang.cindex.TokenKind.PUNCTUATION:
            return "PUNCTUATION:" + s


def analyze_identifier(tokens):
    for t in tokens:
        s = t.spelling
        if t.kind == clang.cindex.TokenKind.IDENTIFIER:
            return "IDENTIFIER:" + s


def analyze_literal(tokens):
    for t in tokens:
        s = t.spelling
        if t.kind == clang.cindex.TokenKind.LITERAL:
            return "LITERAL:" + s


def analyze_function_call(tokens):
    for i in range(len(tokens) - 1):
        current = tokens[i]
        if current.kind == clang.cindex.TokenKind.IDENTIFIER:
            nxt = tokens[i + 1]
            if nxt.kind == clang.cindex.TokenKind.PUNCTUATION and nxt.spelling == "(":
                return "FUNC_CALL:" + current.spelling + nxt.spelling


def analyze_function_def(tokens):
    for i in range(len(tokens) - 2):
        current = tokens[i]
        if current.kind != clang.cindex.TokenKind.KEYWORD and current.spelling not in keywords:
            res = analyze_func_def_name(tokens, [], i)
            if res is not None:
                return res
        else:
            seen_keywords = [current]
            for j in range(i + 1, len(tokens) - 2):
                current = tokens[j]
                if current.kind == clang.cindex.TokenKind.KEYWORD or current.spelling in keywords:
                    seen_keywords.append(current)
                else:
                    return analyze_func_def_name(tokens, seen_keywords, j)
    return None


def analyze_func_def_name(tokens, keywords, i):
    name = tokens[i]
    if name.kind == clang.cindex.TokenKind.IDENTIFIER:
        brace = tokens[i + 1]
        if brace.kind == clang.cindex.TokenKind.PUNCTUATION and brace.spelling == "(":
            ret = "FUNC_DEF:"
            for key in keywords:
                ret += key.spelling + " "
            ret += name.spelling + brace.spelling
            return ret
    return None


def analyze(source_code):
    tu = idx.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', source_code)], options=0)
    tokens = [t for t in tu.get_tokens(extent=tu.cursor.extent)]

    res = analyze_function_def(tokens)
    if res is not None:
        return res
    res = analyze_keyword(tokens)
    if res is not None:
        return res
    res = analyze_function_call(tokens)
    if res is not None:
        return res
    res = analyze_punctation(tokens)
    if res is not None:
        return res
    res = analyze_identifier(tokens)
    if res is not None:
        return res
    res = analyze_literal(tokens)
    if res is not None:
        return res
    return "UNKNOWN:" + source_code
