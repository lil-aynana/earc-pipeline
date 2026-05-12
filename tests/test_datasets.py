import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.datasets import load_hotpotqa

print("=" * 60)
print("TEST — HotpotQA Dataset Loader")
print("=" * 60)

examples = load_hotpotqa(split_size=10)

print(f"\nLoaded {len(examples)} examples\n")

example = examples[0]

print("QUESTION:")
print(example["question"])

print("\nANSWER:")
print(example["answer"])

print("\nCONTEXT DOCUMENTS:")

for i, doc in enumerate(example["context_docs"][:2]):
    print(f"\nDocument {i+1}")
    print(f"Title: {doc['title']}")
    print(f"Text:  {doc['text'][:200]}...")
