""" Default generation plugin for prompts.
"""
from typing import Type, Dict, Any

import openai
from steamship import Block, SteamshipError, Tag, DocTag, File, Steamship
from steamship.data import TagKind, GenerationTag, TagValue
from steamship.invocable import Config, InvocableResponse, create_handler
from steamship.plugin.tagger import Tagger
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest


class PromptGenerationPlugin(Tagger):
    """Default plugin for generating text based on a prompt."""

    class PromptGenerationPluginConfig(Config):
        openai_api_key : str
        max_words : int
        temperature : float
        model: str


    def config_cls(self) -> Type[Config]:
        return self.PromptGenerationPluginConfig

    config: PromptGenerationPluginConfig

    def __init__(self, client: Steamship = None, config: Dict[str, Any] = None):
        super().__init__(client, config)
        openai.api_key = self.config.openai_api_key

    def _generate_text_for(self, text_prompt: str) -> str:
        """Call the API to generate the next section of text."""
        completion = openai.Completion.create(engine=self.config.model, prompt=text_prompt, temperature=self.config.temperature, max_tokens=self.config.max_words)
        return completion.choices[0].text


    def run(self, request: PluginRequest[BlockAndTagPluginInput]) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the text generator against all Blocks of text.
        """


        request_file = request.data.file
        output = BlockAndTagPluginOutput(file=File.CreateRequest(id=request_file.id), tags=[])
        for block in request.data.file.blocks:
            text = block.text
            generated_text = self._generate_text_for(text)
            tags = [Tag.CreateRequest(kind=TagKind.GENERATION, name=GenerationTag.PROMPT_COMPLETION, value={TagValue.STRING_VALUE:generated_text})]
            output_block = Block.CreateRequest(id=block.id, tags=tags)
            output.file.blocks.append(output_block)

        return InvocableResponse(data=output)

