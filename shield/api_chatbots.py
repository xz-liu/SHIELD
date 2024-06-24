import os
import json
import time
import openai
import anthropic
import langchain

from langchain_community.chat_models import ChatPerplexity, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from parse import args
from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_anthropic import ChatAnthropic

from utils import ModelOutputCache


# https://python.langchain.com/v0.1/docs/integrations/chat/google_generative_ai/

# PPLX_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_ORGANIZATION
class APIModelsWrapper:
    def __init__(self, model_name: str, sleep_time=None):
        if sleep_time is None:
            sleep_time = args.api_model_sleep_time
        if 'sonar' in model_name:
            assert os.getenv('PPLX_API_KEY') is not None, 'PPLX_API_KEY is not set'
            self.model = ChatPerplexity(temperature=0, model=model_name)
        elif model_name.startswith('gpt'):
            assert os.getenv('OPENAI_API_KEY') is not None
            assert os.getenv('OPENAI_ORGANIZATION') is not None
            self.model = ChatOpenAI(temperature=0, api_key=os.getenv('OPENAI_API_KEY'), model
            =model_name, organization=os.getenv('OPENAI_ORGANIZATION'))
        elif model_name.startswith('claude'):
            assert os.getenv('ANTHROPIC_API_KEY') is not None
            self.model = ChatAnthropic(temperature=0, model=model_name)
        elif model_name.startswith('gemini'):
            # if GEMINI_API_KEY is set, use it, or use the GOOGLE API KEY
            assert os.getenv('GEMINI_API_KEY') is not None or os.getenv('GOOGLE_API_KEY') is not None
            if os.getenv('GEMINI_API_KEY') is not None:
                api_key = os.getenv('GEMINI_API_KEY')
            else:
                api_key = os.getenv('GOOGLE_API_KEY')
            self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv('GEMINI_API_KEY'),
                                                temperature=0, convert_system_message_to_human=True)
        else:
            raise ValueError('Model name not recognized')
        self.model_name = model_name
        self.cache = ModelOutputCache(model_name)
        self.sleep_time = sleep_time
        self.gemini_backups = 0

    def invoke(self, prompt_str):
        if prompt_str in self.cache:
            return self.cache[prompt_str].content
        try:
            resp = self.model.invoke(prompt_str)

        except openai.RateLimitError:
            print('Rate limit exceeded. Sleeping for 60 seconds.')
            time.sleep(60)
            resp = self.model.invoke(prompt_str)
        except anthropic.InternalServerError:
            print('Internal server error. Sleeping for 60 seconds.')
            time.sleep(60)
            resp = self.model.invoke(prompt_str)
        # anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Output blocked by content filtering policy'}}
        except anthropic.BadRequestError:
            print('Bad request error. ')
            resp = AIMessage(content='I am sorry. [BAD_REQUEST_ERROR] Output blocked by content filtering policy.',
                             role='ai')
        except BaseException as e:
            # if Gemini
            if self.model_name.startswith('gemini'):
                print('Gemini API error. Trying again with another Google API')
                self.gemini_backups += 1
                if self.gemini_backups > 2:
                    # throw  error
                    raise RuntimeError("fuck, gemini api used up")
                api_key = os.getenv(f'GOOGLE_API_KEY_BACKUP{self.gemini_backups}')
                self.model = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key,
                                                    temperature=0)
                resp = self.model.invoke(prompt_str)

            else:
                raise RuntimeError("Unknown Error", e)


        self.cache[prompt_str] = resp
        time.sleep(self.sleep_time)
        return resp.content

    def __call__(self, msg):
        return self.invoke(msg)

    def batch(self, msgs):
        return [self.invoke(msg) for msg in msgs]


class APIModelTokenizerDummy:
    def _convert_msg_like(self, msg_list):
        new_msgs = []
        for msg in msg_list:
            if msg['role'] == 'user':
                new_msgs.append(('human', msg['content']))
            elif msg['role'] == 'system':
                new_msgs.append(('system', msg['content']))
            elif msg['role'] == 'assistant':
                new_msgs.append(('ai', msg['content']))
            else:
                raise ValueError(msg)

        return new_msgs

    def apply_chat_template(self, msg_list, *args, **kwargs):
        chat_template = ChatPromptTemplate.from_messages(self._convert_msg_like(msg_list))
        return chat_template.format()



if __name__ == '__main__':
    wrapper = APIModelsWrapper('gpt-3.5-turbo')
    print(wrapper.invoke('Hello world!'))
    print(wrapper.invoke('Hello!'))
