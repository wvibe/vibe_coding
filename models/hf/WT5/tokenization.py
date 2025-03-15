from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast


class WT5Tokenizer(T5Tokenizer):
    """
    Wrapper class for the T5 tokenizer that ensures compatibility with the WT5 model.
    This tokenizer uses sentencepiece for tokenization and works exactly like the
    original T5 tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the WT5Tokenizer.

        This is a wrapper around the original T5Tokenizer from HuggingFace.
        All parameters are passed directly to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.model_type = "wt5"


class WT5TokenizerFast(T5TokenizerFast):
    """
    Wrapper class for the T5TokenizerFast that ensures compatibility with the WT5 model.
    This is the fast version of the tokenizer that uses the Rust implementation of sentencepiece.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the WT5TokenizerFast.

        This is a wrapper around the original T5TokenizerFast from HuggingFace.
        All parameters are passed directly to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.model_type = "wt5"