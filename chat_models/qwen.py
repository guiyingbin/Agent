import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union, Generator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from http import HTTPStatus
from dashscope import Generation
import json
import requests
from langchain.chat_models.base import BaseChatModel, _generate_from_stream
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain.schema.output import ChatGenerationChunk
from langchain.pydantic_v1 import Field, SecretStr, root_validator

from langchain.schema import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
)
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    ChatMessageChunk,
    HumanMessageChunk,
)

"""
参考LangChain中Baichuan模型写
"""


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """
    将Message类转为Dict，用于输给模型
    :param message:
    :return:
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """
    将得到的结果
    :param _dict:
    :return:
    """
    message_dict = _dict["message"]
    role = message_dict["role"]
    if role == "user":
        return HumanMessage(content=message_dict["content"])
    elif role == "assistant":
        return AIMessage(content=message_dict.get("content", "") or "")
    elif role == "system":
        return SystemMessage(content=message_dict.get("content", "") or "")
    else:
        return ChatMessage(content=message_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    message_dict = _dict["message"]
    role = message_dict["role"]
    content = message_dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


class ChatQWen(BaseChatModel):
    dashscope_api_key: str
    """DashScope key"""
    generator = Generation()
    """Qwen Generator"""
    streaming: bool = False
    """Whether to stream the results or not."""
    request_timeout: int = 60
    """request timeout for chat http requests"""

    model_name = "Qwen"
    """model name of Baichuan, default is `Baichuan2-53B`."""
    temperature: float = 1.0
    """What sampling temperature to use."""
    top_k: int = 50
    """What search sampling control to use."""
    top_p: float = 0.5
    """What probability mass to use."""
    enable_search: bool = False
    """Whether to use search enhance, default is False."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Baichuan API."""
        normal_params = {
            "model_name": self.model_name,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "enable_search": self.enable_search,
        }

        return {**normal_params, **self.model_kwargs}
    @property
    def model(self):
        assert self.model_name in ["qwen_turbo", "qwen_plus"]
        return Generation.Models.qwen_turbo if self.model_name=="qwen_turbo" else Generation.Models.qwen_plus
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            """
            生成对话结果
            :param messages:
            :param stop:
            :param run_manager:
            :param kwargs:
            :return:
            """
            if self.streaming:
                stream_iter = self._stream(
                    messages=messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return _generate_from_stream(stream_iter)

            res = self._chat(messages, **kwargs)

            if res.get("status_code") != 200:
                raise ValueError(f"Error from Baichuan api response: {res}")

            return self._create_chat_result(res)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式输出
        :param messages:
        :param stop:
        :param run_manager:
        :param kwargs:
        :return:
        """
        res = self._chat(messages, **kwargs)

        default_chunk_class = AIMessageChunk
        for response in res:
            if response.get("status_code") != 200:
                raise ValueError(f"Error from Qwen api response: {response}")

            data = response["output"]["choices"]
            for m in data:
                chunk = _convert_delta_to_message_chunk(m, default_chunk_class)
                default_chunk_class = chunk.__class__
                yield ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> Generator:
        """
        采用DashScope的API来call对应的服务
        :param messages:
        :param kwargs:
        :return:
        """
        if self.dashscope_api_key is None:
            raise ValueError("Qwen API key is not set.")

        messages = [_convert_message_to_dict(m) for m in messages]

        responses = self.generator.call(
            self.model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            stream=self.streaming,
            api_key=self.dashscope_api_key,
            incremental_output=self.streaming,  # get streaming output incrementally
            enable_search=self.enable_search,
            temperature=self.temperature,

        )
        return responses

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for m in response["output"]["choices"]:
            message = _convert_dict_to_message(m)
            gen = ChatGeneration(message=message)
            generations.append(gen)

        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "qwen-chat"


if __name__ == '__main__':
    import yaml

    with open("E:\LocalRepository\LLM\Agent\config.yaml", "r") as file:
        data = yaml.load(file.read(), Loader=yaml.FullLoader)
    api_key = data.get("api_key")
    chatmodel = ChatQWen(dashscope_api_key=api_key, streaming=data.get("streaming"),
                         model_name=data.get("model_name"))
    messages = [ChatMessage(role="system", content="你是一个资深饮食推荐师，名字叫运满满，擅长推荐"),
                HumanMessage(content="我身体虚弱，这一周应该吃什么？"),
                AIMessage(content="你是因为导致虚弱的？"),
                HumanMessage(content="我是因为感冒导致虚弱的,我想吃西红柿炒鸡蛋可以吗？")]
    print(chatmodel.invoke(messages).content)
