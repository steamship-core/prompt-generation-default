"""Default generation plugin for prompts."""
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import Field
from steamship import Steamship, Tag
from steamship.data import GenerationTag, TagKind, TagValueKey
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
        openai_api_key: str = Field("", description="An openAI API key to use. If left default, will use Steamship's API key.")
        max_words: int = Field(description="The maximum number of words to generate per request")
        model: Optional[str] = Field("text-davinci-002", description="The OpenAI model to use.  Can be a pre-existing fine-tuned model.")
        temperature: Optional[float] = Field(0.4, description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.")
        top_p: Optional[int] = Field(1, description="Controls the nucleus sampling, where the model considers the results of the tokens with top_p probability mass. Values between 0-1.")
        n_completions: Optional[int] = Field(1, description="How many completions to generate for each prompt.")
        echo: Optional[bool] = Field(False, description= "Echo back the prompt in addition to the completion")
        stop: Optional[str] = Field("", description="Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence. Value is comma separated string version of the sequence.")
        presence_penalty: Optional[int] = Field(0, description="Control how likely the model will reuse words. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.")
        frequency_penalty: Optional[int] = Field(0, description="Control how likely the model will reuse words. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Number between -2.0 and 2.0.")
        best_of: Optional[int] = Field(1, description="Generates best_of completions server-side and returns the \"best\" (the one with the highest log probability per token).")

        def __init__(self, **kwargs):
            kwargs["stop"] = (
                kwargs["stop"].split(",")
                if kwargs["stop"] is not None and isinstance(kwargs["stop"], str)
                else kwargs["stop"]
            )
            super().__init__(**kwargs)

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
        openai.api_key = self.config.openai_api_key

    def _complete_text(
        self, text_prompt: str, suffix: Optional[str] = None, user: Optional[str] = None
    ) -> List[str]:
        """Call the API to generate the next section of text."""
        completion = openai.Completion.create(
            prompt=text_prompt,
            suffix=suffix,
            user=user or "",
            max_tokens=self.config.max_words,
            engine=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            n=self.config.n_completions,
            echo=self.config.echo,
            stop=self.config.stop,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            best_of=self.config.best_of,
        )
        return [choice.text for choice in completion.choices]

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the text generator against all Blocks of text."""

        file = request.data.file
        for block in request.data.file.blocks:
            generated_texts = self._complete_text(block.text)
            for generated_text in generated_texts:
                block.tags.append(
                    Tag(
                        kind=TagKind.GENERATION,
                        name=GenerationTag.PROMPT_COMPLETION,
                        value={TagValueKey.STRING_VALUE: generated_text},
                    )
                )

        return InvocableResponse(data=BlockAndTagPluginOutput(file=file))
