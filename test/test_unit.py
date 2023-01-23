import json
from pathlib import Path

from steamship import File, Block, TaskState
from steamship.data import TagKind, GenerationTag, TagValueKey
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
    assert (len(tags) == config.get("best_of", 1))
    for tag in tags:
        assert (tag.kind == TagKind.GENERATION)
        assert (tag.name == GenerationTag.PROMPT_COMPLETION)
        assert TagValueKey.STRING_VALUE.value is not None
        assert isinstance(TagValueKey.STRING_VALUE.value, str)