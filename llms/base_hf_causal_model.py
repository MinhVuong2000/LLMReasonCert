from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model_path", type=str, help="HUGGING FACE MODEL or model path"
        )
        parser.add_argument(
            "--max_new_tokens", type=int, help="max length", default=512
        )
        parser.add_argument(
            "--top_k", default=None, type=int, help="Generate params: top-k sampling"
        )
        parser.add_argument(
            "--top_p", default=1.0, type=float, help="Generate params: nucleus sampling"
        )
        parser.add_argument(
            "--num_return_sequences",
            default=1,
            type=int,
            help="The number of returned sequences",
        )
        parser.add_argument(
            "--batch_size", default=1, type=int, help="batch size for pipeline"
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Generate params: temperature",
        )

        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")

    def __init__(self, args):
        self.args = args

    def prepare_for_inference(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, token=HF_TOKEN, trust_remote_code=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=self.DTYPE.get(self.args.dtype, None),
            load_in_8bit=self.args.quant == "8bit",
            load_in_4bit=self.args.quant == "4bit",
            trust_remote_code=True,
        )
        self.generator = pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer
        )
        # self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        if self.args.top_k:
            outputs = self.generator(
                llm_input,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_return_sequences=self.args.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                batch_size=self.args.batch_size,
            )
        else:
            outputs = self.generator(
                llm_input,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens,
                num_return_sequences=self.args.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                batch_size=self.args.batch_size,
            )
        if self.args.num_return_sequences == 1:
            return outputs[0]["generated_text"]  # type: ignore
        return [out["generated_text"] for out in outputs]
