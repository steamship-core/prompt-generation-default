"""Default generation plugin for prompts."""
import json
import logging
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import Field
from steamship import Steamship, SteamshipError, Tag
from steamship.data import GenerationTag, TagKind, TagValueKey
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.outputs.plugin_output import OperationType, OperationUnit, UsageReport
from steamship.plugin.request import PluginRequest
from steamship.plugin.tagger import Tagger
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from utils import max_tokens_for_prompt

VALID_MODELS_FOR_BILLING = [
    'text-ada-001',
    'text-babbage-001',
    'text-curie-001',
    'text-davinci-002',
    'text-davinci-003',
    'code-cushman-001',
    'gpt-3.5-turbo',
    'gpt-4'
]

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
        # Load original api key before it is read from TOML, so we know to restrict models for billing
        original_api_key = config['openai_api_key']
        super().__init__(client, config, context)
        openai.api_key = self.config.openai_api_key

        if original_api_key == "" and self.config.model not in VALID_MODELS_FOR_BILLING:
            raise SteamshipError(
                f"This plugin cannot be used with model {self.config.model} while using Steamship's API key. Valid models are {VALID_MODELS_FOR_BILLING}")

        if self.config.model.startswith("gpt-3.5-turbo"):
            self.llm = openai.ChatCompletion
        else:
            self.llm = openai.Completion

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.llm.create(**kwargs)

        result = _completion_with_retry(**kwargs)
        logging.info("Retry statistics: " + json.dumps(_completion_with_retry.retry.statistics))
        return result

    @property
    def _default_params(self) -> Dict[str, Any]:
        common_default_params = {
            "max_tokens": self.config.max_words,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "n": self.config.n_completions,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "request_timeout": self.config.request_timeout,
        }

        completion_default_params = {
            "engine": self.config.model,
            "best_of": self.config.best_of,
            "echo": self.config.echo,
        }

        chat_default_params = {
            "model": self.config.model,
        }
        if self.llm == openai.ChatCompletion:
            return {**common_default_params, **chat_default_params}
        else:
            return {**common_default_params, **completion_default_params}

    def _complete_text(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        suffix: Optional[str] = None,
        user: Optional[str] = None,
    ) -> (List[List[str]], Dict[str, int]):

        if len(prompts) != 1:
            if self.config.max_words == -1:
                raise ValueError("max_words set to -1 not supported for multiple inputs.")
            if self.llm == openai.ChatCompletion:
                raise ValueError("chat completion not supported for multiple inputs.")
        if self.config.max_words == -1:
            self.config.max_words = max_tokens_for_prompt(prompts[0], self.config.model)

        if stop is not None:
            if self.config.stop:
                raise ValueError("`stop` found in both the input and default configuration.")
        else:
            stop = self.config.stop

        stop = stop.split(",") if stop is not None and stop != "" else None
        logging.info(f"Making OpenAI generation call on behalf of user with id: {user}")
        """Call the API to generate the next section of text."""

        if self.llm == openai.Completion:
            invocation_params = {
                "prompt": prompts,
                "suffix": suffix,
                "user": user or "",
                "stop": stop,
                **self._default_params,  # Note: stops are not invocation params just yet
            }
        else:
            invocation_params = {
                "messages": [{"role": "user", "content": prompts[0]}],
                "user": user or "",
                "stop": stop,
                **self._default_params,  # Note: stops are not invocation params just yet
            }
        completion = self.completion_with_retry(
            **invocation_params,
        )
        result = []
        for i in range(0, len(completion.choices), self.config.n_completions):
            result.append(
                [
                    choice.text if self.llm == openai.Completion else choice.message.content
                    for choice in completion.choices[i : i + self.config.n_completions]
                ]
            )
        usage = completion["usage"]
        usage["completion_id"] = completion["id"]

        usage_reports = [
            UsageReport(
                operation_type=OperationType.RUN,
                operation_unit=OperationUnit.PROMPT_TOKENS,
                operation_amount=usage["prompt_tokens"],
                audit_id=usage["completion_id"],
            ),
            UsageReport(
                operation_type=OperationType.RUN,
                operation_unit=OperationUnit.SAMPLED_TOKENS,
                operation_amount=usage["completion_tokens"],
                audit_id=usage["completion_id"],
            ),
        ]
        return result, usage_reports

    @staticmethod
    def _flagged(responses: List[List[str]]) -> bool:
        input_text = "\n\n".join([text for sublist in responses for text in sublist])
        moderation = openai.Moderation.create(input=input_text)
        return moderation["results"][0]["flagged"]

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the text generator against all Blocks of text."""

        file = request.data.file
        prompts = [block.text for block in file.blocks]
        user_id = self.context.user_id if self.context is not None else "testing"

        if self._flagged([prompts]):
            raise SteamshipError("Sorry, this content is flagged as inappropriate by OpenAI.")

        generated_texts, usage_reports = self._complete_text(prompts=prompts, user=user_id)
        if self.config.moderate_output and self._flagged(generated_texts):
            raise SteamshipError(
                "At least one of the responses was flagged as inappropriate by OpenAI. "
                "You may disable this behavior in the plugin by setting moderate_output=False "
                "in the PluginInstance config."
            )
        for block, options in zip(file.blocks, generated_texts):
            for option in options:
                block.tags.append(
                    Tag(
                        kind=TagKind.GENERATION,
                        name=GenerationTag.PROMPT_COMPLETION,
                        value={TagValueKey.STRING_VALUE: option},
                    )
                )

        return InvocableResponse(data=BlockAndTagPluginOutput(file=file, usage=usage_reports))
