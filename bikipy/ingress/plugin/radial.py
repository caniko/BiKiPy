from functools import cached_property

from bikipy.ingress.plugin.base import BasePlugin


class PluginRadial(BasePlugin):
    @cached_property
    def groups(self):

