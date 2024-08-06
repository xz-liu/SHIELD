from vllm import LLM, SamplingParams

import utils
from models import *
from eval import *
from defense import *
from api_chatbots import APIModelsWrapper, APIModelTokenizerDummy

from agent import CopyrightAgent, get_copyright_tool, make_copyright_agent

from log import dump_gen
from huggingface_hub import login

login(os.getenv('HF_TOKEN'))


def eval_lcs(args, wrapper, dataset):
    if args.backend == "vllm":
        outputs_wo_lp, outputs_w_lp = wrapper.run_batched_inference(dataset, max_sent_num=args.max_dataset_num)
    else:
        outputs_wo_lp, outputs_w_lp = wrapper.run_inference(dataset, max_sent_num=args.max_dataset_num)
    eval_lcs_with_data(args.prompt_type, outputs_wo_lp, outputs_w_lp)
    dump_gen(args, outputs_wo_lp, outputs_w_lp)


def eval_contain(args, wrapper, dataset):
    if args.backend == "vllm":
        outputs_wo_lp, outputs_w_lp = wrapper.run_batched_inference(dataset, max_sent_num=args.max_dataset_num)
    else:
        outputs_wo_lp, outputs_w_lp = wrapper.run_inference(dataset, max_sent_num=args.max_dataset_num)
    eval_contain_with_data(outputs_wo_lp, outputs_w_lp)
    dump_gen(args, outputs_wo_lp, outputs_w_lp)


def load_hf_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, device_map='auto',
                                                    attn_implementation="flash_attention_2",
                                                    torch_dtype=get_dtype(args.dtype))
    model.eval()
    return model


class CachedVLLMWrapper:
    def __init__(self, vllm_model, model_id):
        self.vllm_model = vllm_model
        self.model_id = model_id
        self.cache = utils.ModelOutputCache(model_id)
        self.cache_with_ngrams={}

    def _dwc(self, ngram):
        if ngram is None:
            return self.cache
        else:
            if ngram not in self.cache_with_ngrams:
                self.cache_with_ngrams[ngram] = utils.ModelOutputCache(f"{self.model_id}_WITH_NGRAM_{ngram}")
            return self.cache_with_ngrams[ngram]


    def generate(self, prompts, sampling_params=None, ngram=None):
        need_gen_idx = []
        need_gen_prompts = []
        completed = [None for _ in prompts]
        for idx, prompt in enumerate(prompts):
            if prompt not in self._dwc(ngram):
                need_gen_idx.append(idx)
                need_gen_prompts.append(prompt)
            else:
                completed[idx] = self._dwc(ngram)[prompt].content
        if len(need_gen_prompts) > 0:
            need_gen_outputs = self.vllm_model.generate(need_gen_prompts, sampling_params)
            for idx, output in zip(need_gen_idx, need_gen_outputs):
                self._dwc(ngram)[prompts[idx]] = output.outputs[0].text
                completed[idx] = output.outputs[0].text
        return completed


def run_open_model(args):
    dataset = get_dataset(args.dataset, args.prompt_type, args.jailbreak, args.jailbreak_num)
    print("run open model")
    if args.ngram_dataset != 'none':
        ngram_dataset = get_dataset(args.ngram_dataset, args.prompt_type, args.jailbreak, args.jailbreak_num, args.poem_num)
    else:
        ngram_dataset = dataset
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)

    if args.eval_type == "lcs" or args.eval_type == "contain":
        assert args.backend == "vllm", "LCS and Contain evaluation only support VLLM backend."

        if args.defense_type == 'ngram':
            ngram_model = load_or_train_ngram_model(tokenizer, device, args, ngram_dataset)
            vllm_model = LLM(model=args.hf_model_id,
                             dtype=get_dtype(args.dtype))
            vllm_model= CachedVLLMWrapper(vllm_model, args.hf_model_id)
            wrapper = NgramDefenseWrapper(args, vllm_model, ngram_model, tokenizer)
        elif args.defense_type == 'plain':
            vllm_model = LLM(model=args.hf_model_id,
                             dtype=get_dtype(args.dtype))
            vllm_model= CachedVLLMWrapper(vllm_model, args.hf_model_id)
            wrapper = NgramDefenseWrapper(args, vllm_model, None, tokenizer)
        elif args.defense_type == 'agent':
            if args.agent_use_ngram:
                agent_kwargs = dict(ngram_dataset=ngram_dataset, ngram_tokenizer=tokenizer, tokenizer=tokenizer)
            else:
                agent_kwargs = dict(tokenizer=tokenizer)
            agent = make_copyright_agent(args.hf_model_id, agent_kwargs=agent_kwargs)
            wrapper = AgentDefenseWrapper(agent, tokenizer)
        else:
            raise ValueError(f"Defense type {args.defense_type} not supported.")
        
        if args.eval_type == "lcs":
            eval_lcs(args, wrapper, dataset)
        
        if args.eval_type == "contain":
            eval_contain(args, wrapper, dataset)

        del wrapper
        del dataset


def run_api_model(args):
    '''
    Run the API model.
    sample args: --api_model yes --api_model_name gpt-3.5-turbo --defense_type agent
    Important note! for API models, the defense type should be agent or plain
    To save money, run both agent and plain defense in the same run is prohibited
    For plain defense, the output_w_lp will be empty
    For agent defense, the output_wo_lp will be empty
    The api_model_name can be found in the model cards of the API model
    For openai follow https://platform.openai.com/docs/models
        also set OPENAI_API_KEY, OPENAI_ORGANIZATION in the environment variables
    For perplexity.ai follow https://docs.perplexity.ai/docs/model-cards
        also set PPLX_API_KEY in the environment variables
    For claude follow https://docs.anthropic.com/en/docs/models-overview
        also set ANTHROPIC_API_KEY in the environment variables
    '''
    tokenizer_dummy = APIModelTokenizerDummy()

    if args.agent_use_ngram:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        if args.ngram_dataset != 'none':
            ngram_dataset = get_dataset(args.ngram_dataset, args.prompt_type, args.jailbreak, args.jailbreak_num)
        else:
            ngram_dataset = get_dataset(args.dataset, args.prompt_type, args.jailbreak, args.jailbreak_num)
        agent_kwargs = dict(ngram_dataset=ngram_dataset, ngram_tokenizer=tokenizer, tokenizer=tokenizer_dummy)
    else:
        agent_kwargs = dict(tokenizer=tokenizer_dummy)

    agent = make_copyright_agent(args.api_model_name, args.api_model, agent_kwargs=agent_kwargs)
    assert args.defense_type in ['agent', 'plain'], "API model only supports agent and plain defense types."
    wrapper = AgentDefenseWrapper(agent, tokenizer_dummy, run_before_defense=args.defense_type == 'plain',
                                  run_after_defense=args.defense_type == 'agent')
    with agent_apply_chat_template(agent, False):
        dataset = get_dataset(args.dataset, args.prompt_type, args.jailbreak, args.jailbreak_num)
        if args.eval_type == "lcs":
            eval_lcs(args, wrapper, dataset)
        elif args.eval_type == "contain":
            eval_contain(args, wrapper, dataset)


def main():
    args.eval_type= "lcs"
    if args.api_model:
        run_api_model(args)
    else:
        run_open_model(args) 

if __name__ == "__main__":
    main()
