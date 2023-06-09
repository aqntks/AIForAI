
from typing import Any

from langchain.tools import tool, Tool
from langchain.utilities import PythonREPL


@tool
def save_huggingface_model(save_path: str, model: Any) -> str:
    """useful for when you need to save huggingface model to a specific path"""

    model.save_pretrained(save_path)

    return f"{save_path} : Model saved successfully."


def save_huggingface_model_tool() -> Tool:
    """
    return save_hugging_face Tool

    :return: Tool
    """
    return Tool(
            name="save_huggingface_model",
            func=save_huggingface_model,
            description="useful for when you need to save huggingface model to a specific path"
    )


@tool
def load_tokenizer(query: str) -> Any:
    """useful for when you need to load tokenizer"""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(query)

    return tokenizer


def load_tokenizer_tool() -> Tool:
    """
    return load_tokenizer Tool

    :return: Tool
    """
    return Tool(
            name="load_tokenizer",
            func=load_tokenizer,
            description="useful for when you need to load tokenizer"
    )


@tool
def load_data_collator(query: str) -> Any:
    """useful for when you need to load data collator"""

    from transformers import (
        DefaultDataCollator,
        DataCollatorWithPadding,
        DataCollatorForSOP,
        DataCollatorForLanguageModeling,
        DataCollatorForTokenClassification,
        DataCollatorForSeq2Seq,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForWholeWordMask,
    )

    if query.lower() == "padding":
        data_collator = DataCollatorWithPadding
    elif query.lower() == "sop":
        data_collator = DataCollatorForSOP
    elif query.lower() == "languagemodeling":
        data_collator = DataCollatorForLanguageModeling
    elif query.lower() == "tokenclassification":
        data_collator = DataCollatorForTokenClassification
    elif query.lower() == "seq2seq":
        data_collator = DataCollatorForSeq2Seq
    elif query.lower() == "permutationlanguagemodeling":
        data_collator = DataCollatorForPermutationLanguageModeling
    elif query.lower() == "wholewordmask":
        data_collator = DataCollatorForWholeWordMask
    else:
        data_collator = DefaultDataCollator

    return data_collator


def load_data_collator_tool() -> Tool:
    """
    return load_data_collator Tool

    :return: Tool
    """
    return Tool(
            name="load_data_collator",
            func=load_data_collator,
            description="useful for when you need to load data collator"
    )


@tool
def load_training_arguments(query: str, args: dict) -> Any:
    """useful for when you need to load training arguments"""

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=query,
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_eval_batch_size"],
        num_train_epochs=args["num_train_epochs"],
        weight_decay=args["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return training_args


def load_training_arguments_tool() -> Tool:
    """
    return load_training_arguments Tool

    :return: Tool
    """
    return Tool(
            name="load_training_arguments",
            func=load_training_arguments,
            description="useful for when you need to load training arguments"
    )


@tool
def train_model(model: Any, training_args: Any, dataset: Any, tokenizer: Any, collator: Any) -> str:
    """useful for when you need to train model"""

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()

    return f"Model saved successfully."


def train_model_tool() -> Tool:
    """
    return train_model Tool

    :return: Tool
    """
    return Tool(
            name="train_model",
            func=train_model,
            description="useful for when you need to train model"
    )


def python_tool() -> Tool:
    """
    return python Tool

    :return: Tool
    """

    python_repl = PythonREPL()

    return Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run
    )
