"""Default generation plugin for prompts."""
from typing import Any, Dict, List, Optional, Type

import openai
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
        openai_api_key: str
        max_words: int
        model: Optional[str] = "text-davinci-003"
        temperature: Optional[float] = 0.4
        top_p: Optional[int] = 1
        n_completions: Optional[int] = 1
        echo: Optional[bool] = False
        stop: Optional[List[str]] = None
        presence_penalty: Optional[int] = 0
        frequency_penalty: Optional[int] = 0
        best_of: Optional[int] = 1

        def __init__(self, **kwargs):
            kwargs["stop"] = (
                kwargs["stop"].split(",")
                if kwargs["stop"] is not None and isinstance(kwargs["stop"], str)
                else kwargs["stop"]
            )
            super().__init__(**kwargs)

    def config_cls(self) -> Type[Config]:
        return self.PromptGenerationPluginConfig

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
