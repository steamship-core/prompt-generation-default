import tiktoken


def get_num_tokens(text: str, model: str) -> int:
    """Calculate num tokens with tiktoken package."""
    encoder = "gpt2"
    if model in ("text-davinci-003", "text-davinci-002"):
        encoder = "p50k_base"
    if model.startswith("code"):
        encoder = "p50k_base"
    # create a GPT-3 encoder instance
    enc = tiktoken.get_encoding(encoder)
    # encode the text using the GPT-3 encoder
    tokenized_text = enc.encode(text)
    # calculate the number of tokens in the encoded text
    return len(tokenized_text)


MODEL_TO_MAX_CONTEXT_SIZE = {
    "text-davinci-003": 4097,
    "text-curie-001": 2048,
    "text-babbage-001": 2048,
    "text-ada-001": 2048,
    "code-davinci-002": 8000,
    "code-cushman-001": 2048,
}


def model_to_max_context_size(model: str) -> int:
    return MODEL_TO_MAX_CONTEXT_SIZE.get(model, 4097)


def max_tokens_for_prompt(prompt: str, model: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a prompt."""
    num_tokens = get_num_tokens(prompt, model)
    max_size = model_to_max_context_size(model)
    return max_size - num_tokens
