from jinja2 import Template
from json import load
from os import listdir
from os.path import getsize
from requests import request, RequestException
from subprocess import Popen, PIPE, run
from threading import Lock
import typing as t


__all__ = [
    'LLaMaCPP',
    'LLMS',
]


with open('/opt/llms/index.json', 'r') as _f:
    LLMS = load(_f)


class LLaMaCPP:

    def __init__(self):
        self._model_name = None
        self._process = None
        self._readers = 0
        self._read_lock = Lock()
        self._write_lock = Lock()

    def _add_reader(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def _remove_reader(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def set_model(self, model_name: str) -> None:
        if model_name not in self.list_available_models():
            raise Exception(f"Model {model_name} not found")
        with self._write_lock:
            self._model_name = model_name

    def load_model(self, print_log: bool = False, seed: int = None, threads: int = None, kv_cache_type: t.Optional[t.Literal['f16', 'bf16', 'q8_0', 'q5_0', 'q4_0']] = None, context: int = None, temperature: float = None, top_p: float = None, top_k: int = None, min_p: float = None) -> None:
        if self.process_is_alive():
            raise Exception("A model is already loaded. Use stop() before loading a new model.")
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
        with self._write_lock:
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
            if print_log:
                stdout = None
                stderr = None
            else:
                stdout = PIPE
                stderr = PIPE
            self._process = Popen(command, stdout=stdout, stderr=stderr, text=True)
        return None

    def apply_chat_template(self, conversation: t.List[t.Dict[str, str]], enable_thinking: bool = False) -> str:
        short_name = self.short_model_name(self._model_name)
        chat_template: str = LLMS[short_name]['chat_template']
        template = Template(chat_template)
        options: t.Dict[str, t.Any] = {
            'messages': conversation,
            'tools': [],
            'add_generation_prompt': True,
            'enable_thinking': False,
        }
        if LLMS[short_name]['thinking']:
            if LLMS[short_name]['optional_thinking']:
                options['enable_thinking'] = enable_thinking
            else:
                options['enable_thinking'] = True
        else:
            options['enable_thinking'] = False
        return template.render(**options)

    def generate(self, prompt: t.Union[str, t.List[t.Dict[str, str]]], enable_thinking: bool = False, temperature: float = None, top_k: int = None, top_p: float = None, min_p: float = None, n_predict: int = None, grammar: str = None, seed: int = None) -> str:  # type: ignore
        if isinstance(prompt, list):
            prompt = self.apply_chat_template(prompt, enable_thinking)
        json_data: t.Dict[str, t.Any] = {
            'prompt': prompt,
        }
        if temperature is not None:
            json_data['temperature'] = temperature
        if top_k is not None:
            json_data['top_k'] = top_k
        if top_p is not None:
            json_data['top_p'] = top_p
        if min_p is not None:
            json_data['min_p'] = min_p
        if n_predict is not None:
            json_data['n_predict'] = n_predict
        if grammar is not None:
            json_data['grammar'] = grammar
        if seed is not None:
            json_data['seed'] = seed
        self._add_reader()
        try:
            req = request('POST', 'http://127.0.0.1:8432/completion', json=json_data)
            if req.status_code != 200:
                raise Exception(req.text)
            json_return = req.json()
            return json_return['content']
        finally:
            self._remove_reader()

    def process_is_alive(self) -> bool:  # type: ignore
        self._add_reader()
        try:
            if self._process is None:
                return False
            return self._process.poll() is None
        finally:
            self._remove_reader()

    def is_loading(self) -> bool:  # type: ignore
        self._add_reader()
        try:
            req = request('GET', 'http://127.0.0.1:8432/health')
            return req.status_code == 503
        except RequestException:
            return False
        finally:
            self._remove_reader()

    def is_running(self) -> bool:  # type: ignore
        self._add_reader()
        try:
            req = request('GET', 'http://127.0.0.1:8432/health')
            return req.status_code == 200
        except RequestException:
            return False
        finally:
            self._remove_reader()

    def has_error(self) -> bool:  # type: ignore
        self._add_reader()
        try:
            req = request('GET', 'http://127.0.0.1:8432/health')
            return req.status_code not in [200, 503]
        except RequestException:
            return True
        finally:
            self._remove_reader()

    def stop(self) -> None:
        with self._write_lock:
            if self._process is None:
                return None
            self._process.terminate()
            return None

    def kill(self):
        with self._write_lock:
            if self._process is None:
                return None
            self._process.kill()
            return None

    def get_system_message(self) -> t.List[t.Dict[str, str]]:
        short_name = self.short_model_name(self._model_name)
        system_message = LLMS[short_name]['system_message']
        if system_message == '':
            return []
        return [{'role': 'system', 'content': system_message}]

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
        for model in sorted(LLMS.keys(), key=lambda x: len(x) , reverse=True):
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
    return min(int(free_vram / vram_per_layer), layers)



def check_free_vram() -> int:
    """
    Checks the amount of free VRAM on the GPU

    :return: The amount of free VRAM in MB
    """
    nvidia_smi = run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=PIPE, text=True)
    if nvidia_smi.returncode != 0:
        raise Exception(nvidia_smi.stderr)
    return int(nvidia_smi.stdout)
