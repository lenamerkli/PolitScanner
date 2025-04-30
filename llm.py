import typing as t
from json import load
from os import listdir
from os.path import getsize
from subprocess import Popen, PIPE, run


with open('/opt/llms/index.json', 'r') as _f:
    LLMS = load(_f)


class LLaMaCPP:

    def __init__(self):
        self._model_name = None
        self._process = None

    def set_model(self, model_name: str) -> None:
        if model_name not in self.list_available_models():
            raise Exception(f"Model {model_name} not found")
        self._model_name = model_name

    def load_model(self, seed: int = None, threads: int = None, kv_cache_type: t.Optional[t.Literal['f16', 'bf16', 'q8_0', 'q5_0', 'q4_0']] = None, context: int = None, temperature: float = None, top_p: float = None, top_k: int = None, min_p: float = None) -> None:
        if self._model_name is None:
            raise Exception("Model not set")
        short_name = self.short_model_name(self._model_name)
        if short_name is None:
            raise Exception(f"Model {self._model_name} not found")
        if seed is None:
            seed = -1
        if threads is None:
            threads = 16
        if kv_cache_type is None:
            kv_cache_type = 'q8_0'
        context = min_none(context, LLMS[short_name]['context'])
        if temperature is None:
            temperature = LLMS[short_name]['sampling']['temperature']
        if top_p is None:
            top_p = LLMS[short_name]['sampling']['top_p']
        if top_k is None:
            top_k = LLMS[short_name]['sampling']['top_k']
        if min_p is None:
            min_p = LLMS[short_name]['sampling']['min_p']
        offload_layers = calculate_offload_layers(self._model_name, short_name)
        print(f"Loading model {self._model_name} with {offload_layers} layers offloaded")
        command = [
            '/opt/llama.cpp/bin/llama-server',
            '--threads', str(threads),
            '--ctx-size', str(context),
            '--flash-attn',
            '--no-escape',
            '--cache-type-k', kv_cache_type,
            '--cache-type-v', kv_cache_type,
            '--mlock',
            '--n-gpu-layers', str(offload_layers),
            '--model', f'/opt/llms/{self._model_name}',
            '--seed', str(seed),
            '--temp', str(temperature),
            '--top-k', str(top_k),
            '--top-p', str(top_p),
            '--min-p', str(min_p),
            '--host', '127.0.0.1',
            '--port', '8432',
            '--alias', short_name,
        ]
        self._process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
        return None

    def is_loading_or_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def stop(self) -> None:
        if self._process is None:
            return None
        self._process.terminate()
        return None

    def kill(self):
        if self._process is None:
            return None
        self._process.kill()
        return None


    @staticmethod
    def list_available_models() -> t.List[str]:
        directory_list = listdir('/opt/llms/')
        model_list = []
        for entry in directory_list:
            if entry.endswith('.gguf') and LLaMaCPP.short_model_name(entry) is not None:
                model_list.append(entry)
        return model_list

    @staticmethod
    def short_model_name(model_name: str) -> t.Optional[str]:
        for model in LLMS:
            if model_name.startswith(model):
                return model
        return None


def min_none(a: t.Any, b: t.Any) -> t.Any:
    """
    Returns the minimum of two values, or the single value if one of them is None.

    :param a: First value
    :param b: Second value
    :return: The minimum of a and b, or a/b if one of them is None
    """
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def calculate_offload_layers(model_name: str, short_model_name: str) -> int:
    """
    Calculates the number of layers to offload

    :param model_name: The name of the model
    :param short_model_name: The short name of the model
    :return: The number of layers to offload
    """
    free_vram = check_free_vram() - 1000
    llm_size = getsize(f"/opt/llms/{model_name}") / (1024 ** 2) * 1.1
    layers = LLMS[short_model_name]['layers']
    vram_per_layer = llm_size / layers
    return int(free_vram / vram_per_layer)



def check_free_vram() -> int:
    """
    Checks the amount of free VRAM on the GPU

    :return: The amount of free VRAM in MB
    """
    nvidia_smi = run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=PIPE, text=True)
    if nvidia_smi.returncode != 0:
        raise Exception(nvidia_smi.stderr)
    return int(nvidia_smi.stdout)
