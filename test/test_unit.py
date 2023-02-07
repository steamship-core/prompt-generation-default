import json
from pathlib import Path

from steamship import Block, File, TaskState
from steamship.data import GenerationTag, TagKind, TagValueKey
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import PromptGenerationPlugin


def test_tagger():
    config = json.load(Path("config.json").open())

    tagger = PromptGenerationPlugin(config=config)
    content = Path("data/roses.txt").open().read()
    file = File(id="foo", blocks=[Block(text=content, tags=[])])
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = tagger.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None
    assert response.data.file is not None
    assert response.data.file.blocks is not None
    assert len(response.data.file.blocks) == 1

    tags = response.data.file.blocks[0].tags
    assert len(tags) == config.get("best_of", 1)
    for tag in tags:
        assert tag.kind == TagKind.GENERATION
        assert tag.name == GenerationTag.PROMPT_COMPLETION
        tag_value = tag.value[TagValueKey.STRING_VALUE]
        assert tag_value is not None
        assert isinstance(tag_value, str)


def test_tagger_multiblock():
    config = json.load(Path("config.json").open())
    config['n_completions'] = 3
    config['best_of'] = config['n_completions']
    tagger = PromptGenerationPlugin(config=config)
    file = File(id="foo", blocks=[Block(text="Let's count: one two three"), Block(text="The primary colors are: red blue")])
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = tagger.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None
    assert response.data.file is not None
    assert response.data.file.blocks is not None
    assert len(response.data.file.blocks) == 2

    assert len(response.data.file.blocks[0].tags) == config['n_completions']
    first_block_completion = response.data.file.blocks[0].tags[0].value[TagValueKey.STRING_VALUE]
    assert "four" in first_block_completion

    assert len(response.data.file.blocks[1].tags) == config['n_completions']
    second_block_completion = response.data.file.blocks[1].tags[0].value[TagValueKey.STRING_VALUE]
    assert "yellow" in second_block_completion
