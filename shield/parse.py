import argparse


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["huggingface", "vllm"]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size in batched inference.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bsnc",
        choices=["leetcode", "txtbooks", "passages", "bsc", "bsnc", "bsmc", "ssrl", "bep"]
    )
    parser.add_argument(
        "--ngram_dataset",
        type=str,
        default="none",
        choices=["none", "leetcode", "txtbooks", "passages", "bsc", "bsnc", "bsmc", "ssrl", "bep"],
        help="Dataset to train ngram model. Set to 'none' to use the same dataset as the main model."
    )
    parser.add_argument(
        "--agent_precheck",
        type=str2bool,
        default=True,
        help="agent_precheck.",
    )
    parser.add_argument(
        "--agent_postcheck",
        type=str2bool,
        default=False,
        help="agent_postcheck.",
    )
    parser.add_argument(
        "--dump_gen",
        type=str2bool,
        default=True,
        help="Dump generation.",
    )
    parser.add_argument(
        "--jailbreak",
        type=str,
        default="no",
        choices=["no", "general", "copyright"],
        help="Type of jailbreaks to reinforce prompts.",
    )
    parser.add_argument(
        "--jailbreak_num",
        type=int,
        default=30,
        help="Num of new variants for each original prompt. -1 means all.",
    )
    parser.add_argument(
        "--ngram_context_size",
        type=int,
        default=10,
        help="The 'n' in ngram.",
    )
    parser.add_argument(
        "--retrain_ngram",
        type=str2bool,
        default=True,
        help="Retrain NGram model.",
    )
    parser.add_argument(
        "--ngram_epoch",
        type=int,
        default=10,
        help="Training epochs for NGram model.",
    )
    parser.add_argument(
        "--poem_num",
        type=int,
        default=10,
        help="Mixed number of poems.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="none",
        choices=["none", "a", "b", "c"],
        help="The type of used naive prompts. none means eval ppl, a or b means eval lcs, c means eval contain",
    )
    parser.add_argument(
        "--max_dataset_num",
        type=int,
        default=100000,
        help="Maximmum number of dataset sentences to verify lcs, contain, and ppl.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--overwrite_copyright_status",
        type=str,
        choices=["P", "C", 'None'],
        default='None',
    )
    parser.add_argument(
        '--multiple_rounds',
        type=int,
        default=1,
        help='Number of rounds for multiple round inference.'
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and ngram hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--use_cache",
        type=str2bool,
        default=True,
    )
    # user_lower_cache

    parser.add_argument(
        "--use_lower_cache",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["bf16", "fp16"],
        help="Model precsion.",
    )
    parser.add_argument(
        '--defense_type',
        type=str,
        default='ngram',
        choices=['ngram', 'agent', 'plain'],
        help='defense type')
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='not save results')
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache',
        help='cache directory'
    )
    parser.add_argument(
        '--api_model',
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        '--api_model_sleep_time',
        type=float,
        default=3.0,
    )
    parser.add_argument(
        '--api_model_name',
        type=str,
        default='gpt-3.5-turbo',
        help='api model name')
    parser.add_argument(
        '--agent_use_pplx',
        type=str2bool,
        default=True,
        help='use pplx for agent')
    parser.add_argument(
        '--agent_use_ngram',
        type=str2bool,
        default=True,
        help='use ngram for agent')

    args = parser.parse_args()

    if args.prompt_type == "a" or args.prompt_type == "b":
        args.eval_type = "lcs"
    elif args.prompt_type == "c":
        args.eval_type = "contain"
    elif args.prompt_type == "none":
        args.eval_type = "ppl"
    else:
        raise Exception("no such prompt type")

    return args


args = parse_args()
