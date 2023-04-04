"""Test assemblyai-s2t-blockifier via integration tests."""
import json
from pathlib import Path

from steamship import Block, File, Steamship, TaskState
from steamship.data import GenerationTag, TagKind, TagValueKey

GENERATOR_HANDLE = "gpt-3"
ENVIRONMENT = "prod"


def test_generator():
    """Test the AssemblyAI Blockifier via an integration test."""
    with Steamship.temporary_workspace(profile=ENVIRONMENT) as steamship:
        plugin_instance = steamship.use_plugin(
            plugin_handle=GENERATOR_HANDLE,
            config=json.load(Path("config.json").open()),
        )

        content = Path("data/roses.txt").open().read()
        file = File.create(steamship, blocks=[Block(text=content, tags=[])])

        blockify_task = file.tag(plugin_instance=plugin_instance.handle)
        blockify_task.wait(max_timeout_s=3600, retry_delay_s=1)

        assert blockify_task.state == TaskState.succeeded
        file = blockify_task.output.file

        assert file is not None
        assert file.blocks is not None
        assert len(file.blocks) == 1
        tags = file.blocks[0].tags
        assert len(tags) == json.load(Path("config.json").open()).get("best_of", 1)
        for tag in tags:
            assert tag.kind == TagKind.GENERATION
            assert tag.name == GenerationTag.PROMPT_COMPLETION
            assert TagValueKey.STRING_VALUE.value is not None
            assert isinstance(TagValueKey.STRING_VALUE.value, str)
