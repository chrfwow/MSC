import re
from typing import List

from counterfactuals2.AbstractTokenizer import AbstractTokenizer
from counterfactuals2.language import Language


class RegexTokenizer(AbstractTokenizer):
    cpp_tokens: List[str] = ["", " ", "\n", "(", ")", "{", "}", "[", "]", ";", "\"", "<<", "<", ">>", ">",
                             "::", ".", ",", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "!=", "==", "=",
                             "||", "|", "&&", "&", "~", "'", "->", "true", "false"]
    cpp_keywords: List[str] = ["do", "while", "for", "break", "return", "if", "else", "int", "bool", "double", "float",
                               "long", "const", "unsigned", "switch", "struct", "nullptr", "free", "malloc", "case",
                               "len", ]

    java_tokens: List[str] = ["", " ", "\n", "(", ")", "{", "}", "[", "]", ";", "\"", "<<", "<", ">>", ">",
                              "::", ".", ",", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "!=", "==", "=",
                              "||", "|", "&&", "&", "'", "->", "true", "false"]
    java_keywords: List[str] = ["return", "do", "while", "for", "break", "return", "if", "else", "int", "boolean",
                                "double", "float", "long", "final", "static", "switch", "null", "case", "String"]

    tokens = []
    keywords = []

    def __init__(self, language: Language):
        super().__init__(language)
        if language == Language.Cpp:
            self.tokens = self.cpp_tokens
            self.keywords = self.cpp_keywords
        elif language == Language.Java:
            self.tokens = self.java_tokens
            self.keywords = self.java_keywords
        else:
            print("unknown language", language)
            return

        for i in range(10):
            self.tokens.append(str(i))

    def tokenize(self, source_code: str) -> (int, List[str]):
        """Returns a tuple containing the number of words in the document, and a list of all available words,
        including words not in the document"""
        escaped_dict: List[str] = []
        for token in self.tokens:
            if token != "" and token != " " and token != "\n":
                escaped_dict.append(re.escape(token))

        escaped_keywords: List[str] = []
        for word in self.keywords:
            escaped_keywords.append(word + "\\s+")

        delimiter = "|".join(escaped_dict) + "|" + "|".join(escaped_keywords)
        delimiter = "(\\s+|\\-?[0-9]*[\\.]?[0-9]+f?|\\-?[0-9]+l?|" + delimiter + ")"

        doc_parts = re.split(delimiter, source_code)
        doc_parts = [word.strip() for word in doc_parts]
        doc_parts = list(filter(lambda a: not re.match(r"\s+", a), doc_parts))
        doc_parts = list(filter(None, doc_parts))
        doc_parts_set = {*doc_parts}
        additions = [*self.tokens, *self.keywords]

        directory_without_doc_parts: List[str] = []

        for token in additions:
            if token not in doc_parts_set:
                directory_without_doc_parts.append(token)

        return len(doc_parts), [*doc_parts, *directory_without_doc_parts]
