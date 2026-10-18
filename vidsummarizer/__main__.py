"""Entry point for 'python -m vidsummarizer' (source-checkout runs).

The built executable (zipapp) generates its own __main__ stub calling
vidsummarizer.cli:main; this one covers running from a checkout.
"""

from .cli import main

main()
