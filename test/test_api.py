import json

from steamship.data import TagKind, GenerationTag, TagValue
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput

from src.api import PromptGenerationPlugin
from steamship import File, Block, TaskState, DocTag, Steamship
from steamship.plugin.request import PluginRequest

from test import TEST_DATA


def _read_test_file(filename: str) -> str:
    file_path = TEST_DATA / filename
    with open(file_path, 'r') as f:
        return f.read()


def test_tagger():
    with open('../config.json') as config_file:
        config = json.load(config_file)

    tagger = PromptGenerationPlugin(config=config)
    content = _read_test_file('roses.txt')
    file = File(id="foo", blocks=[Block(text=content, tags=[])])
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = tagger.run(request)

    assert (response.status.state is TaskState.succeeded)
    assert (response.data is not None)
    assert (response.data.file is not None)
    assert (response.data.file.blocks is not None)
    assert (len(response.data.file.blocks) == 1)

    tags = response.data.file.blocks[0].tags
    assert len(tags) == 1
    tag = tags[0]
    assert tag.kind == TagKind.GENERATION
    assert tag.name == GenerationTag.PROMPT_COMPLETION
    new_text = tag.value.get(TagValue.STRING_VALUE.value, '')
    print(new_text)
    assert len(new_text) > 0

def test_deployed_tagger():
    client = Steamship()
    with open('../config.json') as config_file:
        config = json.load(config_file)

    tagger = client.use_plugin(plugin_handle='prompt-generation-default', config=config)
    content = _read_test_file('roses.txt')
    file = File.create(client, blocks=[Block.CreateRequest(text=content)])
    file.tag(plugin_instance=tagger.handle).wait()
    file = file.refresh()
    response = tagger.run(request)

    assert (len(file.blocks) == 1)

    tags = response.data.file.blocks[0].tags
    assert len(tags) == 1
    tag = tags[0]
    assert tag.kind == TagKind.GENERATION
    assert tag.name == GenerationTag.PROMPT_COMPLETION
    new_text = tag.value.get(TagValue.STRING_VALUE.value, '')
    print(new_text)
    assert len(new_text) > 0
