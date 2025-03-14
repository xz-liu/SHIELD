
# [SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation](https://arxiv.org/abs/2406.12975)

![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/xz-liu/SHIELD/blob/master/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arxiv-2406.12975-red)](https://arxiv.org/abs/2406.12975)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defenses targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defenses to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanisms significantly reduce the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. 



## Dataset

The BS-NC and BEP datasets are made public in this repository. The copyrighted datasets are not made public due to legal restrictions. 

The jailbreak templates are collected from [Liu et al's Google Docs](https://docs.google.com/spreadsheets/d/1s8LtY4nM0C-ufFgM2CC6hRzH6FI592BI/edit?usp=sharing&ouid=115869021074995548314&rtpof=true&sd=true). Please refer to [their paper](https://arxiv.org/abs/2305.13860) for details.



### Update on dataset

We have received numerous inquiries regarding the sharing of copyrighted datasets. However, we are not aware of any precedent for publicly releasing such datasets in North America. To mitigate any potential legal risks, we will be consulting with the relevant university departments. This process is expected to be lengthy, but if you would like to stay informed, please feel free to [send me an email](mailto:xiaoze@purdue.edu).

## Setup Environment

We provide a requirements.txt file to install the required dependencies.

```angular2html
pip install -r requirements.txt
```



## Setup API Key


Please refer to the respective API documentation to get the API key. Once you have the API key, please set it in the environment variable. 


To allow agent search, please set the following ppplx API key.

```angular2html
export PPLX_API_KEY=<API_KEY>
```

For Claude and Gemini, please set

```angular2html
export ANTHROPIC_API_KEY=<API_KEY>
export GOOGLE_API_KEY=<API_KEY>
```

For the OpenAI API key, please set the organization key as well.
```angular2html
export OPENAI_API_KEY=<API_KEY>
export OPENAI_ORGANIZATION=<API_KEY>
```

To access the gated models on huggingface, please login with [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) or set `HF_TOKEN`.

## Run

For open-source models, use the following command to run the code.

```angular2html
python main.py  --max_dataset_num 100 --batch_size <BATCH_SIZE> --dtype fp16 --defense_type <DEFENSE_TYPE> --prompt_type <PROMPT_TYPE>  --jailbreak_num -1 --hf_model_id <HF_MODEL_ID> --dataset <DATASET> --jailbreak <JAILBREAK> 
```

For API-based models, use the following command to run the code.

```angular2html 
python main.py --max_dataset_num 100 --batch_size 1 --defense_type <DEFENSE_TYPE> --prompt_type <PROMPT_TYPE>  --jailbreak_num -1 --api_model yes --api_model_name <API_MODEL_NAME> --api_model_sleep_time <API_MODEL_SLEEP_TIME> --dataset <DATASET> --jailbreak <JAILBREAK> 
```

Explanation of the arguments:


| Argument | Explanation |
| --- | --- |
| max_dataset_num | The number of samples to evaluate. Set to 100 for all titles to be evaluated. |
| batch_size | The batch size for the model. Please adjust according to the GPU memory. |
| dtype | The data type for the model. FP16 or BF16, default is FP16. |
| temperature | The temperature of the model. The default is 0. |
| defense_type | The defense type to be used. Select from 'plain' for no defense, 'agent' for agent-based defense, and 'ngram' for n-gram-based defense. |
| prompt_type | The prompt type to be used. Select from 'a' for prefix probing, 'b' for direct probing|
| api_model | Set to 'yes' for API-based models, 'no' for open-source models. |
| hf_model_id | The Hugging Face model ID is to be used for open-source models. If defense_type is 'plain', and api_model is 'yes', the argument is not required. If defense_type is 'agent' or 'ngram', the argument is required. The model ID is also used as the tokenizer of n-gram defense. |
| api_model_name | The API model name to be used for API-based models.  |
| api_model_sleep_time | The sleep time for the API model, allows for not exceeding the API rate limit. |
| dataset | The dataset to be used. Select from 'bsnc' and 'bep'. |
| jailbreak | Set to 'general' for jailbreak, 'no' for no jailbreak. |
| jailbreak_num | The number of jailbreaks to be used. Set to -1 for all jailbreaks. |
| agent_precheck/agent_postcheck | The agent precheck (check input prompt) and postcheck (check input prompt plus generated text) to be used. Set to 'yes' for agent precheck and postcheck, 'no' for no agent precheck and postcheck. 
| overwrite_copyright_status | You can disable the copyright status verifier and overwrite all the input as 'C' (copyrighted) or 'P' (public domain). This is used to analysis the effectiveness of the few-shot prompts. In common usage, leave it to default value.|



To see the detailed arguments, you can run the following command.
    
```angular2html
python main.py --help
```

# Citation

Please cite our [paper](https://arxiv.org/abs/2406.12975) if you find the repository helpful.

```
@misc{liu2024shieldevaluationdefensestrategies,
      title={SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation}, 
      author={Xiaoze Liu and Ting Sun and Tianyang Xu and Feijie Wu and Cunxiang Wang and Xiaoqian Wang and Jing Gao},
      year={2024},
      eprint={2406.12975},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12975}, 
}
```
