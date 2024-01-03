"""Default generation plugin for prompts."""
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import Field
from steamship import Steamship, SteamshipError, Tag
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.tagger import Tagger


class PromptGenerationPlugin(Tagger):
    """Default plugin for generating text based on a prompt.

    Notes
    -----
    * The parameters logit_bias, logprobs and stream are not supported.
    """

    class PromptGenerationPluginConfig(Config):
        openai_api_key: str = Field(
            "",
            description="An openAI API key to use. If left default, will use Steamship's API key.",
        )
        max_words: int = Field(description="The maximum number of words to generate per request")
        model: Optional[str] = Field(
            "text-davinci-003",
            description="The OpenAI model to use.  Can be a pre-existing fine-tuned model.",
        )
        temperature: Optional[float] = Field(
            0.4,
            description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.",
        )
        top_p: Optional[int] = Field(
            1,
            description="Controls the nucleus sampling, where the model considers the results of the tokens with top_p probability mass. Values between 0-1.",
        )
        n_completions: Optional[int] = Field(
            1, description="How many completions to generate for each prompt."
        )
        echo: Optional[bool] = Field(
            False, description="Echo back the prompt in addition to the completion"
        )
        stop: Optional[str] = Field(
            "",
            description="Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence. Value is comma separated string version of the sequence.",
        )
        presence_penalty: Optional[int] = Field(
            0,
            description="Control how likely the model will reuse words. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.",
        )
        frequency_penalty: Optional[int] = Field(
            0,
            description="Control how likely the model will reuse words. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Number between -2.0 and 2.0.",
        )
        best_of: Optional[int] = Field(
            1,
            description='Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token).',
        )
        moderate_output: bool = Field(
            True,
            description="Pass the generated output back through OpenAI's moderation endpoint and throw an exception if flagged.",
        )
        max_retries: int = Field(
            8, description="Maximum number of retries to make when generating."
        )
        request_timeout: Optional[float] = Field(
            600,
            description="Timeout for requests to OpenAI completion API. Default is 600 seconds.",
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.PromptGenerationPluginConfig

    config: PromptGenerationPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):
        super().__init__(client, config, context)
        raise SteamshipError("This plugin has been deprecated due to OpenAI's deprecation of its underlying models.  Please use the gpt-4 Generator plugin instead.")

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        raise SteamshipError("This plugin has been deprecated due to OpenAI's deprecation of its underlying models.  Please use the gpt-4 Generator plugin instead.")
