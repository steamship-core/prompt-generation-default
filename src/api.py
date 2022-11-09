"""Example Steamship Tagger Plugin.

In Steamship, **Taggers** are responsible emitting tags that describe the **Steamship Block Format**.
"""
from typing import Type

from steamship import Block, SteamshipError, Tag, DocTag, File
from steamship.data import TagKind
from steamship.invocable import Config, InvocableResponse, create_handler
from steamship.plugin.tagger import Tagger
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest


def _sentence_tag(start_idx, end_idx: int) -> Tag.CreateRequest:
    return Tag.CreateRequest(
        kind=TagKind.DOCUMENT,
        start_idx=start_idx,
        end_idx=end_idx,
        name=DocTag.SENTENCE,
    )


class TaggerPlugin(Tagger):
    """Example Steamship Tagger Plugin."""

    def run(self, request: PluginRequest[BlockAndTagPluginInput]) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Every plugin implements a `run` function.

        This template plugin does an extremely simple form of text tagging. It generates tags for all sentences
        (based on ANY period that it sees) in the text blocks it sees. It also assumes no repetition in sentences.
        """
        if request is None:
            return InvocableResponse(error=SteamshipError(message="Missing PluginRequest"))

        if request.data is None:
            return InvocableResponse(error=SteamshipError(message="Missing BlockAndTagPluginInput"))

        if request.data.file is None:
            return InvocableResponse(error=SteamshipError(message="Missing `file` field in BlockAndTagPluginInput"))

        request_file = request.data.file
        output = BlockAndTagPluginOutput(file=File.CreateRequest(id=request_file.id), tags=[])
        for block in request.data.file.blocks:
            text = block.text
            # split on '.' to "find" the sentences, then generate text
            sentences = [x.strip() + '.' for x in text.split(".") if len(x) > 0]
            # assume no repetition in sentences
            tags = [_sentence_tag(text.index(s), text.index(s) + len(s)) for s in sentences]
            output_block = Block.CreateRequest(id=block.id, tags=tags)
            output.file.blocks.append(output_block)

        return InvocableResponse(data=output)


handler = create_handler(TaggerPlugin)
