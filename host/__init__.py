"""The host: the program that runs on the machine (`python -m host`).

It owns the chat platform (a channel, channels/<name>/), the users (users.py) and their
sandboxes (sandbox.py): one Linux container per user, each running that user's agent (the `arti`
package, with plugins/). host.py passes messages between the channel and the agents, through
files in each agent's data directory (arti/link.py). The operator's commands are in __main__.py.
"""
