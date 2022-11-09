from steamship.data import TagKind
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput

from api import TaggerPlugin
from steamship import File, Block, TaskState, DocTag
from steamship.plugin.request import PluginRequest

from test import TEST_DATA

WANT_SENTENCES = ["A Poem.", "Roses are red.", "Violets are blue.", "Sugar is sweet, and I love you."]


def _read_test_file(filename: str) -> str:
    file_path = TEST_DATA / filename
    with open(file_path, 'r') as f:
        return f.read()


def test_tagger():
    tagger = TaggerPlugin()
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
    assert (len(tags) == 4)
    for tag in tags:
        assert (tag.kind == TagKind.DOCUMENT)
        assert (tag.name == DocTag.SENTENCE)
        assert (content[tag.start_idx:tag.end_idx] in WANT_SENTENCES)
