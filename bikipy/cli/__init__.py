import click


@click.group
def cli_root():
    pass


from bikipy.cli.ingress import ingress
from bikipy.cli.inspect import inspect
