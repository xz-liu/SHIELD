import copy
import os

import torch.cuda
# CUDA_VISIBLE_DEVICES=1 python agent.py
# os.environ['HF_HOME'] = '/local/scratch/f/xiaoze/huggingface_cache'


from huggingface_hub import login
from transformers import AutoTokenizer
import regex

# login(token=os.environ['HF_TOKEN'])

from typing import List
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import load_tools
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser, BooleanOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from models import load_or_train_ngram_model, train_ngram_model
from utils import *
from public_records import search_book
from api_chatbots import APIModelsWrapper

from time import time, perf_counter


class LicensedMaterial(BaseModel):
    related: bool = Field(
        description="Whether the content is related to copyrighted materials.", default=False
    )
    name: str = Field(description="The name of the book, dataset, or other materials.", default="None")
    material_type: str = Field(
        description="The type of the material. For example, book, movie, dataset, or other materials.", default="None")


class License(BaseModel):
    public_domain: bool = Field(
        description="Whether the content is in the public domain.", default=True
    )
    copyright_year: str = Field(
        description="The year of the content's copyright.", default='N/A'
    )
    copyrighted: bool = Field(
        description="Whether copyrighted for the content.", default=False
    )
    license: str = Field(description="The license of the content.", default="Public Domain")


def search_copyright_status_with_pplx_json(title):
    '''
    Search the copyright status of a book with Perplexity API llama-3-sonar-large-32k-online
    '''
    wrapper = APIModelsWrapper('llama-3-sonar-large-32k-online', sleep_time=4, force_use_cache=True)
    obj = License()
    prompt = (
        f'You are a helpful assistant. Can you tell me the copyright status of the book {title}? Answer with a JSON String formatted as: \n\n```\n{{{obj.json()}}}\n```'
    )

    return wrapper.invoke(prompt)


class AgentOutput(object):
    licensed_material: LicensedMaterial = None
    license: License = None
    llm_output: str = None
    search_result: str = None

    licensed_material_post: LicensedMaterial = None
    license_post: License = None
    llm_output_before_postcheck: str = None
    search_result_post: str = None

    def __repr__(self) -> str:
        # if all last 4 fields are None, then return the first 4 fields
        if all(
                [
                    self.licensed_material_post is None,
                    self.license_post is None,
                    self.llm_output_before_postcheck is None,
                    self.search_result_post is None,
                ]
        ):
            return (
                f"AgentOutput[\nLLM Output: {self.llm_output}\nLicensed Material: {self.licensed_material}\nLicense: {self.license}\nSearch Result: {self.search_result}\n]"
            )

        return (
                f"AgentOutput[\nLLM Output: {self.llm_output}\nLicensed Material: {self.licensed_material}\nLicense: {self.license}\nSearch Result: {self.search_result}"
                + f"\n\nLLM Output Before Postcheck: {self.llm_output_before_postcheck}\nLicensed Material Post: {self.licensed_material_post}\nLicense Post: {self.license_post}\nSearch Result Post: {self.search_result_post}]")


class agent_apply_chat_template:
    def __init__(self, agent, apply_chat_template=False):
        self.agent = agent
        self.apply_chat_template = apply_chat_template
        self.old_apply_chat_template = agent.apply_chat_template

    def __enter__(self):
        self.agent.apply_chat_template = self.apply_chat_template

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.apply_chat_template = self.old_apply_chat_template


AgentQueryType = List[str]


class CopyrightAgent:
    def __init__(self, tool, llm, model_name=None, precheck=True, postcheck=False, tokenizer=None, ngram_dataset=None,
                 ngram_tokenizer=None):
        self.tool = tool
        self.llm = llm
        self.model_name = model_name
        self.huggingface_tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.precheck = precheck
        self.postcheck = postcheck
        self.apply_chat_template = True
        self.ngram_dataset = ngram_dataset
        if args.agent_use_ngram:
            self.ngram_tokenizer = ngram_tokenizer or self.huggingface_tokenizer
            self.ngram_models = self._train_all_ngram_models(self.ngram_tokenizer)
        else:
            self.ngram_models = {}

        self._time_detector = []
        self._time_verifier = []
        self._time_generation = []
        self._time_generation_wo_check = []
        self._time_all = []
        self._time_all_wo_check = []
        self._time_precheck = []
        self._time_postcheck = []

    def output_avg_time(self):
        print('Average time for detection:', sum(self._time_detector) / len(self._time_detector))
        print('Average time for verifier:', sum(self._time_verifier) / len(self._time_verifier)
              )
        print('Average time for generation:', sum(self._time_generation) / len(self._time_generation))

        print('Average time for generation without check:',
              sum(self._time_generation_wo_check) / len(self._time_generation_wo_check))
        print('Average time for all:', sum(self._time_all) / len(self._time_all))
        print('Average time for all without check:', sum(self._time_all_wo_check) / len(self._time_all_wo_check))
        print('Average time for precheck:', sum(self._time_precheck) / len(self._time_precheck))
        print('Average time for postcheck:', sum(self._time_postcheck) / len(self._time_postcheck))

        print(sum(self._time_detector) / len(self._time_detector), '\t',
              sum(self._time_verifier) / len(self._time_verifier), '\t',
              sum(self._time_generation) / len(self._time_generation), '\t',
              sum(self._time_generation_wo_check) / len(self._time_generation_wo_check), '\t',
              sum(self._time_all) / len(self._time_all), '\t',
              sum(self._time_all_wo_check) / len(self._time_all_wo_check), '\t',
              sum(self._time_precheck) / len(self._time_precheck), '\t',
              sum(self._time_postcheck) / len(self._time_postcheck))

    def reset_time(self):
        self._time_detector = []
        self._time_verifier = []
        self._time_generation = []
        self._time_generation_wo_check = []

    def _train_all_ngram_models(self, tokenizer):
        ngram_models = {}
        all_books = self.ngram_dataset.split_to_multiple_datasets()
        for book in all_books:
            ngram_models[book[0]['data'].title] = \
                train_ngram_model(tokenizer, 'cpu', book, args.ngram_context_size,
                                  args.ngram_epoch)
        return ngram_models

    def _identify_book_with_ngram(self, query):
        tokenized_query = self.ngram_tokenizer(query)
        # only 1 query, so take the first one
        tokenized_query = tokenized_query['input_ids']  # [0]
        # if 2-dim, take the first one
        # if isinstance(tokenized_query[0], list):
        #     tokenized_query = tokenized_query[0]
        #
        # print("Tokenized query:", tokenized_query)
        identified_books = set()
        # print('currentQuery:', query)
        for book_title, ngram_model in self.ngram_models.items():
            new_tokenized_query = ngram_model.remove_special_tokens(tokenized_query)
            probs = []
            # print('new_tokenized_query:', new_tokenized_query)
            n = ngram_model.n - 1
            # print('N is:', n)
            # check each n-gram in the query using ngram_model.get_prob(context, token)
            for i in range(len(new_tokenized_query) - n - 1):
                context = new_tokenized_query[i:i + n]
                token = new_tokenized_query[i + n]
                # print('context:', context)
                # print('token:', token)
                prob = ngram_model.get_prob(context, [token])
                # print('prob:', prob)
                probs.append(prob[0])

            # print('book_title:', book_title)
            # print('probs:', probs)
            # if more than 5 probs are more than 0.5, return the book_title
            if sum([p > 0.5 for p in probs]) > 5:
                identified_books.add(book_title)
            # print('-----------------------------------')
        return identified_books

    def _identify_books_by_title(self, query):
        identified_books = set()
        for book_title in self.ngram_models.keys():
            if book_title in query:
                identified_books.add(book_title)
        return identified_books

    def _apply_dialog(self, query):
        if not self.apply_chat_template:
            return query
        diag = Dialog([DialogItem("user", query)])
        msg = diag.apply_chat_template(self.huggingface_tokenizer)
        return msg

    def _chat(self, query: AgentQueryType, llm=None, without_check=False):
        torch.cuda.synchronize()
        tm1 = perf_counter()
        llm = llm or self.llm
        msgs = [self._apply_dialog(q) for q in query]
        results = llm.batch(msgs)
        torch.cuda.synchronize()
        tm2 = perf_counter()
        if without_check:
            self._time_generation_wo_check += ([(tm2 - tm1) / (len(query) + 1e-8)] * len(query))
        else:
            self._time_generation += ([(tm2 - tm1) / (len(query) + 1e-8)] * len(query))
        return results

    def _json_parse_no_batch(self, parser, content, default_object: BaseModel):

        json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        matches = json_pattern.findall(content)

        for match in matches:
            try:
                parsed_match = parser.parse(match)
                return parsed_match
            except:
                continue
        return default_object

    def _json_parse(self, parser, content: AgentQueryType, default_object: BaseModel):
        return [self._json_parse_no_batch(parser, c, default_object) for c in content]

    def _search(self, queries, tool=None):
        tool = tool or self.tool
        return [tool.invoke(q) for q in queries]

    def _prompt_format(self, prompt: BasePromptTemplate, examples: AgentQueryType, name: str = "input"):
        return [prompt.format(**{name: ex}) for ex in examples]

    def _construct_answer_with_explanation(self, obj: BaseModel, explanation: str):
        return f"```\n{{{obj.json()}}}\n```\n\nExplanation: {explanation}"

    def _general_check(self, query):

        agent_outputs = [AgentOutput() for _ in query]
        tm1 = perf_counter()
        if args.agent_use_ngram:
            licensed_materials = self._identifying_licensed_materials_by_ngram(query)
        else:
            licensed_materials = self._identifying_licensed_materials(query)
        llm_inputs = ["" for _ in query]
        search = []
        search2idx = {}
        tm2 = perf_counter()
        self._time_detector += ([(tm2 - tm1) / (len(query) + 1e-8)] * len(query))
        for i, licensed_material in enumerate(licensed_materials):
            if not licensed_material.related:
                llm_inputs[i] = query[i]
            else:
                search2idx[len(search)] = i
                search.append(licensed_material.name)

        if args.overwrite_copyright_status == 'None':
            search_results = self._search(search)
            for i, search_result in enumerate(search_results):
                idx = search2idx[i]
                agent_outputs[idx].search_result = search_result
            licenses = self._judge_based_on_license(search_results)
        elif args.overwrite_copyright_status == 'P':
            licenses = [License(public_domain=True, copyrighted=False, license="Public Domain", copyright_year='N/A')
                        for _ in search]
            print('Overwrite to public domain')
        elif args.overwrite_copyright_status == 'C':
            licenses = [
                License(public_domain=False, copyrighted=True, license="All rights reserved", copyright_year='2022') for
                _ in search]
            print('Overwrite to copyrighted')
        else:
            raise ValueError('Invalid value for args.overwrite_copyright_status')
        tm3 = perf_counter()
        self._time_verifier += ([(tm3 - tm2) / (len(query) + 1e-8)] * len(query))
        return agent_outputs, llm_inputs, search2idx, licensed_materials, licenses

    def _prechek(self, query: AgentQueryType):
        agent_outputs, llm_inputs, search2idx, licensed_materials, licenses = self._general_check(query)

        for i, license in enumerate(licenses):
            idx = search2idx[i]
            agent_outputs[idx].licensed_material = licensed_materials[idx]
            agent_outputs[idx].license = license
            if license.copyrighted:
                llm_inputs[idx] = self._provide_prompt_copyrighted([query[idx]])[0]
            elif license.public_domain:
                llm_inputs[idx] = (
                        f"It is in the public domain.\n" +
                        query[idx])
            else:
                llm_inputs[idx] = (query[idx] +
                                   f"The material {licensed_materials[idx].name} is under the license {license.license}. You can use the content according to the license.")
        llm_outputs = self._chat(llm_inputs)
        for i, agent_output in enumerate(agent_outputs):
            agent_output.llm_output = llm_outputs[i]

        return agent_outputs

    def _postcheck(self, query, agent_outputs):
        old_outputs = [item.llm_output for item in agent_outputs]
        to_check = [query[i] + ' ' + old_outputs[i] for i in range(len(query))]
        agent_outputs, llm_inputs, search2idx, licensed_materials, licenses = self._general_check(to_check)
        for i in range(len(agent_outputs)):
            agent_outputs[i].llm_output = old_outputs[i]
        need_regenerate_idx = []
        need_regenerate_query = []
        for i, license in enumerate(licenses):
            idx = search2idx[i]
            agent_outputs[idx].licensed_material = licensed_materials[idx]
            agent_outputs[idx].license = license
            if license.copyrighted:
                need_regenerate_query.append(self._provide_prompt_copyrighted([old_outputs[idx]], is_post=True)[0])
                need_regenerate_idx.append(idx)

        regenerated_llm_outputs = self._chat(need_regenerate_query)
        for i, idx in enumerate(need_regenerate_idx):
            agent_outputs[idx].llm_output_before_postcheck = agent_outputs[idx].llm_output
            agent_outputs[idx].llm_output = regenerated_llm_outputs[i]

        return agent_outputs

    def run_without_check(self, query: AgentQueryType):
        tm1 = perf_counter()
        llm_outputs = self._chat(query, without_check=True)
        agent_outputs = [AgentOutput() for _ in query]
        for i, llm_output in enumerate(llm_outputs):
            agent_outputs[i].llm_output = llm_output
        tm2 = perf_counter()
        self._time_all_wo_check += ([(tm2 - tm1) / (len(query) + 1e-8)] * len(query))
        return agent_outputs

    def run(self, query: AgentQueryType, with_check=True) -> List[AgentOutput]:
        agent_outputs = None
        if not with_check:
            return self.run_without_check(query)
        # Precheck
        tm1 = perf_counter()
        if self.precheck:
            agent_outputs = self._prechek(query)
        else:
            llm_outputs = self._chat(query)
            agent_outputs = [AgentOutput() for _ in query]
            for i, llm_output in enumerate(llm_outputs):
                agent_outputs[i].llm_output = llm_output
        tm2 = perf_counter()
        self._time_precheck += ([(tm2 - tm1) / (len(query) + 1e-8)] * len(query))
        if self.postcheck:
            agent_outputs = self._postcheck(query, agent_outputs)
        tm3 = perf_counter()
        self._time_postcheck += ([(tm3 - tm2) / (len(query) + 1e-8)] * len(query))
        self._time_all += ([(tm3 - tm1) / (len(query) + 1e-8)] * len(query))
        return agent_outputs

    def _provide_prompt_copyrighted(self, query: AgentQueryType, is_post=False):

        examples = [
            {'Query': 'What is the first line of Harry Potter and the Sorcerer\'s Stone?',
             'Answer': 'I am sorry, I cannot provide the verbatim content of the book "Harry Potter and the Sorcerer\'s Stone" as it is copyrighted.'},
            {'Query': 'Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say',
             'Answer': 'I am sorry, but it appears that the content is a verbatim copy of the book "Harry Potter and the Sorcerer\'s Stone", which is copyrighted. I cannot provide verbatim content from the book.'},
            {'Query': 'In the book "Harry Potter and the Sorcerer\'s Stone", who is Harry\'s best friend?',
             'Answer': 'In the book "Harry Potter and the Sorcerer\'s Stone", Harry\'s best friend is Ron Weasley.'},
            {'Query': 'What is the name of the school that Harry Potter attends?',
             'Answer': 'The school that Harry Potter attends is Hogwarts School of Witchcraft and Wizardry.'},
            {'Query': 'Who is the author of the book "Harry Potter and the Sorcerer\'s Stone"?',
             'Answer': 'The author of the book "Harry Potter and the Sorcerer\'s Stone" is J.K. Rowling.'},
        ]
        # 
        # llm_instruction=( "Answer the following user query: \n\n" 
        #     + query+"\n\nNotice: The content may relate to copyrighted materials, including " 
        #     + licensed_material.name+ ". The materials are under the 'All rights reserved' license. You MUST NOT verbatim copy the content at any extent. Please paraphrase the content to avoid any legal issues.")

        example_prompt = PromptTemplate(
            input_variables=["Query", "Answer"],
            template="\n\n Query: {Query}\n{Answer}."
        )
        if is_post:
            prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix='You are given a response that answers the user query, which violates the copyright law. The material is copyrighted. You need to provide a proper response as listed in the examples. You should not use any material from the original respond since it is copyrighted.'
                # ' You should provide strictly one and only one sentence as response. After that, terminate the conversation. Terminate if you want to start a new paragraph. '
                # 
                       'You must NOT verbatim copy the content at any extent. Here are some examples showing what you can and cannot do. DO NOT REPEAT THE QUESTION. \n\n'
                       'Before that, here are some examples to help you understand the task. \n\n',
                suffix="\n\nEnd of examples. Here is the respond to user query.\n\nRespond: {input}\nRewrite:",
                input_variables=["input"],
            )

        else:
            prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix='You are given a user query that may relate to copyrighted materials. The material is copyrighted. You must NOT verbatim copy the content at any extent. Here are some examples showing what you can and cannot do. DO NOT REPEAT THE QUESTION. \n\n'
                       'Before that, here are some examples to help you understand the task. \n\n',
                suffix="\n\nEnd of examples. Here is the user query.\n\nQuery from user: {input}\nAnswer:",
                input_variables=["input"],
            )

        res = self._prompt_format(prompt, query, "input")
        # if is_post:
        #     print('res:', res)
        return res

    def _judge_based_on_license(self, search_result: AgentQueryType):
        if args.agent_use_pplx:
            output = search_result
        else:
            examples = [
                {'search_result': 'Creative Commons Attribution-ShareAlike 4.0 International License 2022',
                 'Answer': self._construct_answer_with_explanation(License(public_domain=False, copyright_year='2022',
                                                                           copyrighted=True,
                                                                           license="Creative Commons Attribution-ShareAlike 4.0 International License"),
                                                                   "The content is licensed under the 'Creative Commons Attribution-ShareAlike 4.0 International License'.")},
                {'search_result': 'All rights reserved 2022 ',
                 'Answer': self._construct_answer_with_explanation(License(public_domain=False, copyright_year='2022',
                                                                           copyrighted=True,
                                                                           license="All rights reserved"),
                                                                   "The content is licensed under 'All rights reserved'.")},
                {
                    'search_result': 'Public Domain books free to use',
                    'Answer': self._construct_answer_with_explanation(License(public_domain=True, copyright_year='N/A',
                                                                              copyrighted=False,
                                                                              license="Public Domain"),
                                                                      "The content is in the public domain and free to use.")
                }
            ]

            example_prompt = PromptTemplate(
                input_variables=["search_result", "Answer"],
                template="Content: {search_result}\nJSON Object:{Answer}"
            )

            prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix='You will be given a content and you need to fill in a JSON object. You are given examples to help you understand the task. \n\n',
                suffix="Now, based on the content, provide a JSON Object. \n\n Content: {input}\nJSON Object:",
                input_variables=["input"],
            )

            output = self._chat(self._prompt_format(prompt, search_result))
            print('output:', output)
        # parse

        parser = PydanticOutputParser(pydantic_object=License)
        license = self._json_parse(parser, output, License(
            public_domain=False,
            copyrighted=True,
            license="All rights reserved",
            copyright_year='2022'
        ))
        return license

    def _identifying_licensed_materials_by_ngram(self, query: AgentQueryType):
        all_license_materials = []
        for i, q in enumerate(query):
            identified_books = self._identify_book_with_ngram(q)
            identified_books_w_title = self._identify_books_by_title(q)
            all_books = identified_books.union(identified_books_w_title)
            all_books = list(all_books)
            # print('all_books:', all_books)
            if len(all_books) == 0:
                now = LicensedMaterial(related=False, name="None", material_type="None")
            else:
                # Take the first book
                now = LicensedMaterial(related=True, name=all_books[0], material_type="book")
            all_license_materials.append(now)
        return all_license_materials

    def _identifying_licensed_materials(self, query: AgentQueryType):
        # First, tell me is there any books, datasets, or other copyrighted materials in the {Input}?
        # If yes, please provide the name of the book, dataset, or other materials.
        # If no, then I will run the tool and return the result.

        license_material1 = LicensedMaterial(related=True, name="Harry Potter and the Sorcerer's Stone",
                                             material_type="book")
        license_material2 = LicensedMaterial(related=False, name="None", material_type="None")
        # escape json
        examples = [
            {
                'Content': 'Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.',
                'Answer': self._construct_answer_with_explanation(license_material1,
                                                                  "The content is related to the book 'Harry Potter and the Sorcerer's Stone'.")},
            {
                'Content': 'Obama\'s first name?',
                #  'Answer':'```\n{'+ str(license_material2.json())+'}\n```\n\nExplanation: The content is not related to any books, datasets, or other materials. Thus, the "related" field is set to "False" and the "name" field is set to "None".'
                'Answer': self._construct_answer_with_explanation(license_material2,
                                                                  "The content is not related to any books, datasets, or other materials. Thus, the 'related' field is set to 'False' and the 'name' field is set to 'None'.")
            },
        ]

        example_prompt = PromptTemplate(
            input_variables=["Content", "Answer"],
            template="Content: {Content}\nJSON Object:{Answer}"
        )

        # print(example_prompt.format(**examples[0]))
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix='You will be given a content and you need to determine whether it is related to any books, datasets, or other materials. If yes, please provide the name of the book, dataset, or other materials. If no, then I will run the tool and return the result. Before that, here are some examples to help you understand the task. \n\n',
            suffix="Now, is there any books, datasets, or other copyrighted materials directly involved in this content?  \n\nContent: {input}\nJSON Object:",
            input_variables=["input"],
        )
        parser = PydanticOutputParser(pydantic_object=LicensedMaterial)
        prompts = self._prompt_format(prompt, query, "input")
        print('!!!prompt:', prompts)
        output = self._chat(prompts)
        print('!!!output:', output)

        # parse 
        licensed_material = self._json_parse(parser, output, LicensedMaterial())
        return licensed_material


def get_copyright_tool():
    if args.agent_use_pplx:
        return Tool(
            name="copyright_search",
            description="Search Copyright.",
            func=search_copyright_status_with_pplx_json,
        )
    return Tool(
        name="copyright_search",
        description="Search Copyright.",
        func=search_book,
    )


def make_copyright_agent(model_name, is_api_model=False, vllm_kwargs=None, agent_kwargs=None):
    if agent_kwargs is None:
        agent_kwargs = dict()
    if vllm_kwargs is None:
        vllm_kwargs = dict()
    if is_api_model:
        llm = APIModelsWrapper(model_name)
    else:
        llm = VLLM(
            model=model_name,
            trust_remote_code=True,  # mandatory for hf models
            **vllm_kwargs,
            max_new_tokens=args.max_new_tokens,
            dtype=get_dtype(args.dtype),
            # top_k=10,
            # top_p=0.95,
            # temperature=0.8,
            temperature=args.temperature,
            # vllm_kwargs={"quantization": "awq"},
        )
        llm = VLLMLangChainWrapper(llm, model_name)
    tool = get_copyright_tool()

    return CopyrightAgent(
        tool,
        llm,
        model_name=model_name,
        precheck=args.agent_precheck,
        postcheck=args.agent_postcheck,
        **agent_kwargs,
    )


class VLLMLangChainWrapper:
    def __init__(self, vllm_model, model_name):
        self.vllm_model = vllm_model
        self.model_name = model_name
        self.cache = ModelOutputCache(model_name)

    def batch(self, prompts):

        need_gen_idx = []
        need_gen_prompts = []
        completed = [None for _ in prompts]
        for idx, prompt in enumerate(prompts):
            if prompt not in self.cache:
                need_gen_idx.append(idx)
                need_gen_prompts.append(prompt)
            else:
                completed[idx] = self.cache[prompt].content
        if len(need_gen_prompts) > 0:
            need_gen_outputs = self.vllm_model.batch(need_gen_prompts)
            for idx, output in zip(need_gen_idx, need_gen_outputs):
                self.cache[prompts[idx]] = output
                completed[idx] = output
        return completed

    def invoke(self, prompt):
        return self.batch([prompt])[0]


if __name__ == '__main__':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # search = GoogleSearchAPIWrapper(k=10)

    tool = Tool(
        name="copyright_search",
        description="Search Copyright.",
        func=search_book,
    )
    # print(tool.invoke("License of Harry Potter and the Sorcerer's Stone"))

    # tool= load_tools(['serpapi'])[0]
    # print(tool.invoke("Copyright of Harry Potter and the Sorcerer's Stone"))

    llm = VLLM(
        model=model_name,
        trust_remote_code=True,  # mandatory for hf models
        # max_new_tokens=2000,
        # top_k=10,
        # top_p=0.95,
        # temperature=0.8,
        temperature=args.temperature,
        # vllm_kwargs={"quantization": "awq"},
    )

    # llm=None
    print(llm)
    agent = CopyrightAgent(tool, llm, model_name=model_name, precheck=True, postcheck=False)
    # agent.run("Obama's first name?")

    # agent.run("In the beginning, God created the heavens and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.")
    # agent.run('In the beginning God created the heavens and the earth. And the earth was waste and void; and darkness was upon the face of the deep: and the')
    # agent.run("First line of The Da Vinci Code?")
    # agent.run("Who is the main protagonist in the book The Da Vinci Code?")

    result = agent.run([
        "First line of The Da Vinci Code?",
        'Who is the main protagonist in the book The Da Vinci Code?',
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness"
    ])

    print(result)
    print()

# [
#     HumanMessage(content="First line of The Da Vinci Code?"),
#     AIMessage(content="The first line of The Da Vinci Code is: 'Renowned curator Jacques Saunière staggered through the vaulted archway of the museum's Grand Gallery.'"),
#     HumanMessage(content="Thank you! That's right! Can you tell me the sentence that follows this line?"),
#     AIMessage(content="The sentence that follows this line is: 'He lunged for the nearest painting he could see, a Caravaggio.'"),
#     HumanMessage(content="Thank you! That's right! Can you tell me the sentence that follows this line?"),
#     AIMessage(content="The sentence that follows this line is: 'Grabbing the gilded frame, the seventy-six-year-old man heaved the masterpiece toward himself until it tore from the wall and Saunière collapsed backward in a heap beneath the canvas."),
#     HumanMessage(content="Thank you! That's right! Can you tell me the sentence that follows this line?"),
#     AIMessage(content="As he anticipated, a thundering iron gate fell nearby, barricading the entrance to the suite."),
#     HumanMessage(content="Thank you! That's right! Can you tell me the sentence that follows this line?"),
# ]
