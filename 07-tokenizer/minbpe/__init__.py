from .base import Tokenizer
from .basic import BasicTokenizer

# RegexTokenizer needs the third-party `regex` module (pip install regex).
# Import it lazily so BasicTokenizer still works without that dependency.
try:
    from .regex import RegexTokenizer
except ModuleNotFoundError:
    RegexTokenizer = None
