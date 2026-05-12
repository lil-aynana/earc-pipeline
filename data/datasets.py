from datasets import load_dataset


def load_hotpotqa(split_size: int = 500) -> list[dict]:
    """
    Load only N examples from HotpotQA distractor setting.
    """

    print(f"Loading HotpotQA — {split_size} examples only...")

    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=f"validation[:{split_size}]"
    )

    examples = []

    for item in dataset:

        context_docs = []

        for title, sentences in zip(
            item["context"]["title"],
            item["context"]["sentences"]
        ):

            context_docs.append({
                "title": title,
                "text": " ".join(sentences)
            })

        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "context_docs": context_docs
        })

    print(f"HotpotQA loaded: {len(examples)} examples")

    return examples
