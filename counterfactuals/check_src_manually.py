from typing import Tuple

import torch
from transformers import AutoTokenizer, PLBartForSequenceClassification

from common.code_formatter import format_code

defect_detection = "uclanlp/plbart-c-cpp-defect-detection"
base = "uclanlp/plbart-base"
finetuned = "mrm8488/codebert2codebert-finetuned-code-defect-detection"
model_path = defect_detection  # finetuned

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PLBartForSequenceClassification.from_pretrained(model_path, problem_type="multi_label_classification")

num_labels = len(model.config.id2label)
model = PLBartForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,
                                                        problem_type="multi_label_classification")


def classify(document) -> Tuple[bool, float]:
    document = format_code(document, "java")
    inputs = tokenizer(document, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
    labels = torch.sum(
        torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ).to(torch.float)
    loss = model(**inputs, labels=labels).loss

    return labels[0][0] >= .5, float(loss)


print(classify("""
public class Main{
public static void main(String[] args){
    break System.out.println("hello");
}}
""".strip()))

print(classify("""
public class Main{
public static void main(String[] args){
    System.out.println("hello");
}}
""".strip()))

print(classify("""
public class Main{
public static void main(String[] args){
   aldkhjaskhkjash
}}
""".strip()))

print(classify("""
public static void main(String[] args){
    System.out.println("hello");
}}
""".strip()))

print(classify("""
asdasdasd
""".strip()))

print(classify("""
asdasdasd
""".strip()))

print(classify("""
asdasdasd
""".strip()))

print(classify("""
System.out.println("hello world");
""".strip()))

print(classify("""
int a = new "ASD";
""".strip()))

print(classify("""
int a = 26734;
""".strip()))

print(classify("""
std::cout << "Hello, World!" << std::endl;
""".strip()))

print(classify("""
std::cout << "Hello, World! | std::endl;
""".strip()))